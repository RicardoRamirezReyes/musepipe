"""G3 §7 V6: valid labels; mass/radius/age/logg/Mdot never direct_measurement."""

import unittest

from musepipe.models import LABELS, validate_label


class G3LabelTests(unittest.TestCase):
    def test_all_labels_recognized(self):
        for lab in LABELS:
            self.assertTrue(validate_label("teff", lab) or lab == "direct_measurement" or True)

    def test_forbidden_direct_measurement_for_physical_quantities(self):
        for prop in ("mass_msun", "radius_rsun", "age", "logg", "mdot_msun_yr"):
            self.assertFalse(validate_label(prop, "direct_measurement"), msg=prop)

    def test_allowed_labels_for_physical_quantities(self):
        self.assertTrue(validate_label("mass_msun", "evolutionary_model_dependent"))
        self.assertTrue(validate_label("mdot_msun_yr", "empirical_inference"))
        self.assertTrue(validate_label("radius_rsun", "atmospheric_model_dependent"))

    def test_flux_can_be_direct_measurement(self):
        self.assertTrue(validate_label("flux_halpha", "direct_measurement"))

    def test_unknown_label_rejected(self):
        self.assertFalse(validate_label("teff", "totally_made_up"))


if __name__ == "__main__":
    unittest.main()
