import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from astropy.io import fits

from musepipe.reduction.esorex_driver import (
    ReductionError,
    SofEntry,
    RecipePlan,
    build_esorex_command,
    build_inventory,
    build_recipe_plan,
    check_esorex_environment,
    ensure_a1_run_tree,
    parse_params,
    parse_esorex_recipes,
    parse_log_warnings,
    read_products_json,
    validate_sof,
    write_inventory_csv,
    write_sof,
)


def _write_header_fits(path, dpr_type, dpr_catg="CALIB", extra=None):
    header = fits.Header()
    header["HIERARCH ESO DPR TYPE"] = dpr_type
    header["HIERARCH ESO DPR CATG"] = dpr_catg
    header["HIERARCH ESO INS MODE"] = "WFM-AO"
    header["DATE-OBS"] = "2022-08-29T01:02:03"
    header["EXPTIME"] = 10.0
    header["HIERARCH ESO DET WIN1 BINX"] = 1
    header["HIERARCH ESO DET WIN1 BINY"] = 1
    for key, value in (extra or {}).items():
        header[key] = value
    fits.PrimaryHDU(header=header).writeto(path)


class ReductionDriverTests(unittest.TestCase):
    def test_parse_esorex_recipes_extracts_recipe_names(self):
        output = """
          muse_bias          : Bias recipe
          muse_scipost       : Science post-processing
          cr2res_obs_2d      : Other recipe
        """
        self.assertEqual(
            parse_esorex_recipes(output),
            ["muse_bias", "muse_scipost", "cr2res_obs_2d"],
        )

    def test_check_esorex_environment_requires_muse_recipes(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / "esorex"
            exe.write_text("#!/bin/sh\n", encoding="utf-8")
            exe.chmod(0o755)

            def runner(args):
                if args[-1] == "--version":
                    return subprocess.CompletedProcess(args, 0, stdout="EsoRex version 3.13.10", stderr="")
                return subprocess.CompletedProcess(args, 0, stdout="cr2res_obs_2d : recipe", stderr="")

            with self.assertRaises(ReductionError):
                check_esorex_environment(esorex=str(exe), runner=runner)

    def test_check_esorex_environment_accepts_muse_recipes(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / "esorex"
            exe.write_text("#!/bin/sh\n", encoding="utf-8")
            exe.chmod(0o755)

            def runner(args):
                if args[-1] == "--version":
                    return subprocess.CompletedProcess(args, 0, stdout="EsoRex version 3.13.10", stderr="")
                if "--man-page" in args:
                    # The MUSE plugin version comes from a recipe man-page, not the
                    # esorex banner. Mirror the real "<recipe> -- version X" line.
                    return subprocess.CompletedProcess(
                        args, 0, stdout=f"  {args[-1]} -- version 2.8.7\n", stderr=""
                    )
                return subprocess.CompletedProcess(
                    args,
                    0,
                    stdout="muse_bias : recipe\nmuse_scipost : recipe",
                    stderr="",
                )

            env = check_esorex_environment(esorex=str(exe), runner=runner)
            self.assertEqual(env["muse_pipeline"], "2.8.7")
            self.assertIn("muse_scipost", env["muse_recipes"])

    def test_build_inventory_classifies_by_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for idx in range(5):
                path = Path(tmp) / f"bias_{idx}.fits"
                _write_header_fits(path, "BIAS")
                paths.append(path)
            flat = Path(tmp) / "flat.fits"
            _write_header_fits(flat, "FLAT,LAMP")
            paths.append(flat)

            records = build_inventory(paths, checksum=False)
            tags = [record.tag for record in records]
            self.assertEqual(tags.count("BIAS"), 5)
            self.assertEqual(tags.count("FLAT"), 1)
            self.assertTrue(all(record.ins_mode == "WFM-AO" for record in records))

    def test_inventory_preserves_pro_category(self):
        with tempfile.TemporaryDirectory() as tmp:
            master = Path(tmp) / "master_bias.fits"
            _write_header_fits(master, "", "", extra={"HIERARCH ESO PRO CATG": "MASTER_BIAS"})
            record = build_inventory([master], checksum=False)[0]
            self.assertEqual(record.tag, "MASTER_BIAS")
            self.assertEqual(record.pro_catg, "MASTER_BIAS")

    def test_build_recipe_plan_enforces_minimum_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for idx in range(5):
                path = Path(tmp) / f"bias_{idx}.fits"
                _write_header_fits(path, "BIAS")
                paths.append(path)
            records = build_inventory(paths, checksum=False)
            plan = build_recipe_plan("muse_bias", records)
            self.assertEqual(plan.recipe, "muse_bias")
            self.assertEqual(len(plan.entries), 5)
            self.assertTrue(all(entry.tag == "BIAS" for entry in plan.entries))

    def test_build_recipe_plan_uses_static_tables_from_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            std_flux = Path(tmp) / "std_flux.fits"
            extinct = Path(tmp) / "extinct.fits"
            pixtable = Path(tmp) / "pixtable_std.fits"
            _write_header_fits(std_flux, "STATIC", extra={"HIERARCH ESO PRO CATG": "STD_FLUX_TABLE"})
            _write_header_fits(extinct, "STATIC", extra={"HIERARCH ESO PRO CATG": "EXTINCT_TABLE"})
            _write_header_fits(pixtable, "PRODUCT", extra={"HIERARCH ESO PRO CATG": "PIXTABLE_STD"})
            records = build_inventory([std_flux, extinct], checksum=False)
            plan = build_recipe_plan(
                "muse_standard",
                records,
                products={"PIXTABLE_STD": [pixtable]},
            )
            tags = [entry.tag for entry in plan.entries]
            self.assertEqual(tags, ["PIXTABLE_STD", "STD_FLUX_TABLE", "EXTINCT_TABLE"])

    def test_validate_sof_raises_on_header_tag_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bias.fits"
            _write_header_fits(path, "BIAS")
            plan = RecipePlan("muse_bias", (SofEntry(path, "FLAT"),))
            with self.assertRaises(ReductionError):
                validate_sof(plan)

    def test_write_inventory_and_sof(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bias.fits"
            _write_header_fits(path, "BIAS")
            records = build_inventory([path], checksum=False)
            csv_path = Path(tmp) / "raw_inventory.csv"
            write_inventory_csv(records, csv_path)
            self.assertIn("BIAS", csv_path.read_text(encoding="utf-8"))

            sof_path = Path(tmp) / "muse_bias.sof"
            write_sof(RecipePlan("muse_bias", (SofEntry(path, "BIAS"),)), sof_path)
            self.assertEqual(sof_path.read_text(encoding="utf-8").strip(), f"{path} BIAS")

    def test_build_esorex_command_records_only_requested_params(self):
        plan = RecipePlan("muse_scipost", (), params={"save": "cube,individual"})
        command = build_esorex_command(
            plan,
            "scipost.sof",
            output_dir="out",
            log_file="log.txt",
        )
        self.assertIn("--save=cube,individual", command)
        self.assertIn("--output-dir=out", command)
        # Recipe parameters must come after the recipe name; SOF stays last.
        self.assertEqual(command[-1], "scipost.sof")
        self.assertLess(command.index("muse_scipost"), command.index("--save=cube,individual"))

    def test_read_products_json_accepts_string_or_list_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "products.json"
            path.write_text(
                '{"MASTER_BIAS": "bias.fits", "PIXTABLE_OBJECT": ["a.fits", "b.fits"]}',
                encoding="utf-8",
            )
            products = read_products_json(path)
            self.assertEqual(products["MASTER_BIAS"], ["bias.fits"])
            self.assertEqual(products["PIXTABLE_OBJECT"], ["a.fits", "b.fits"])

    def test_parse_params_requires_key_value(self):
        self.assertEqual(parse_params(["save=cube,individual"]), {"save": "cube,individual"})
        with self.assertRaises(ReductionError):
            parse_params(["save"])

    def test_parse_log_warnings_elevates_known_risky_warnings(self):
        parsed = parse_log_warnings(
            "[ WARNING ] missing calibration\n[ WARNING ] harmless\n[ ERROR ] bad\n"
        )
        self.assertEqual(len(parsed["warnings"]), 2)
        self.assertEqual(len(parsed["errors"]), 1)
        self.assertEqual(len(parsed["elevated"]), 1)

    def test_ensure_a1_run_tree_creates_config_and_stage_qc(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = ensure_a1_run_tree(
                tmp,
                raw_data_dir=Path(tmp) / "raw_data",
                adp_reference=None,
                environment={"esorex": "test"},
            )
            self.assertTrue(paths["config_json"].exists())
            self.assertTrue(paths["stage_qc"].exists())
            self.assertTrue(str(paths["stage_qc"]).endswith("runs/ROXs12b_raw/stages/stage00r_qc.json"))


if __name__ == "__main__":
    unittest.main()
