import unittest

from utils.flatten_statement import (
    extract_biomarker_info,
    extract_indication,
    extract_therapy_info,
    flatten_statements,
)


def therapy(name, strategy, therapy_type):
    return {
        "name": name,
        "extensions": [
            {"name": "therapy_strategy", "value": [strategy]},
            {"name": "therapy_type", "value": therapy_type},
        ],
    }


def statement():
    return {
        "id": "stmt:test",
        "description": "Evidence statement",
        "extensions": [
            {
                "name": "indication",
                "value": {"description": "Advanced EGFR-positive lung cancer"},
            }
        ],
        "reportedIn": [
            {
                "documentType": "Regulatory approval",
                "urls": ["https://example.org/label"],
                "extensions": [
                    {
                        "name": "agent",
                        "value": {"id": "agent:org:fda"},
                    },
                    {"name": "publication_date", "value": "2026-08-02"},
                ],
            }
        ],
        "proposition": {
            "conditionQualifier": {"name": "Non-Small Cell Lung Cancer"},
            "objectTherapeutic": therapy(
                "Osimertinib", "EGFR inhibition", "Targeted therapy"
            ),
            "extensions": [
                {
                    "name": "biomarkers",
                    "value": [
                        {
                            "present": True,
                            "subject": {"name": "EGFR exon 19 deletion"},
                        },
                        {
                            "present": False,
                            "subject": {"name": "EGFR T790M"},
                        },
                    ],
                }
            ],
        },
    }


class CurrentApiSchemaTests(unittest.TestCase):
    def test_extracts_indication_extension(self):
        self.assertEqual(
            extract_indication(statement()),
            "Advanced EGFR-positive lung cancer",
        )

    def test_extracts_biomarker_extension(self):
        biomarkers = extract_biomarker_info(statement())
        self.assertEqual(
            biomarkers["list"], ["EGFR exon 19 deletion", "EGFR T790M"]
        )
        self.assertIn("EGFR exon 19 deletion [present]", biomarkers["str"])
        self.assertIn("EGFR T790M [not present]", biomarkers["str"])

    def test_extracts_direct_therapy(self):
        result = extract_therapy_info(statement())["list"]
        self.assertEqual(result["drugList"], ["Osimertinib"])
        self.assertEqual(result["therapy_approach"], "Monotherapy")
        self.assertEqual(result["therapy_strategyList"], ["EGFR inhibition"])

    def test_extracts_combination_therapy(self):
        row = statement()
        row["proposition"]["objectTherapeutic"] = {
            "membershipOperator": "AND",
            "therapies": [
                therapy("Drug A", "Strategy A", "Targeted therapy"),
                therapy("Drug B", "Strategy B", "Immunotherapy"),
            ],
        }

        result = extract_therapy_info(row)["list"]
        self.assertEqual(result["drugList"], ["Drug A", "Drug B"])
        self.assertEqual(result["therapy_approach"], "Combination therapy")

    def test_flattens_current_statement(self):
        summary, row = flatten_statements(statement())
        self.assertEqual(row["indication"], "Advanced EGFR-positive lung cancer")
        self.assertEqual(row["cancer_type"], "Non-Small Cell Lung Cancer")
        self.assertEqual(row["approval_org"], "agent:org:fda")
        self.assertIn("Indication: Advanced EGFR-positive lung cancer", summary)


if __name__ == "__main__":
    unittest.main()
