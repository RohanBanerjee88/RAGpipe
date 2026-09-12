"""Import fidelity, source versioning, and dataset sample boundaries."""

import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from openpyxl import Workbook

from knowledge_store import import_collection, source_sections, token_chunks
from grounding import validate_grounded_answer


class WordTokenizer:
    def __call__(self, text, **kwargs):
        return {"offset_mapping": [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]}


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.env = patch.dict(os.environ, {"FAQ_DATA_DIR": str(self.root / "store")})
        self.env.start()
        self.docs = self.root / "docs"
        self.docs.mkdir()

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def ingest(self):
        return import_collection("lab", self.docs, WordTokenizer())

    def test_idempotent_import_and_changed_source_version(self):
        path = self.docs / "protocol.md"
        path.write_text("# Storage\nKeep samples at -80 C.\n", encoding="utf-8")
        first = self.ingest()["records"][0]
        self.assertEqual(first, self.ingest()["records"][0])
        path.write_text("# Storage\nKeep samples at -20 C.\n", encoding="utf-8")
        second = self.ingest()["records"][0]
        self.assertEqual(first["source_id"], second["source_id"])
        self.assertEqual(second["version"], first["version"] + 1)
        self.assertIsNone(second["scraped_at"])

    def test_failed_reimport_retains_previous_records(self):
        path = self.docs / "faq.json"
        path.write_text('[{"question":"Storage?","answer":"Freezer."}]')
        first = self.ingest()
        path.write_text("invalid JSON")
        second = self.ingest()
        self.assertEqual(first["records"], second["records"])
        self.assertTrue(second["warnings"])

    def test_directory_reimport_removes_deleted_source(self):
        path = self.docs / "notes.txt"
        path.write_text("Keep RNA frozen.")
        self.ingest()
        path.unlink()
        self.assertEqual(self.ingest()["records"], [])

    def test_samples_are_bounded_and_do_not_guess_column_meaning(self):
        path = self.docs / "samples.csv"
        path.write_text("id,x\n" + "\n".join(f"sample{i},{i}" for i in range(20)))
        text = self.ingest()["records"][0]["text"]
        self.assertIn("sample4", text)
        self.assertNotIn("sample5", text)
        self.assertIn("Bounded sample only", text)
        self.assertNotIn("expression", text)

    def test_workbook_formulas_are_never_evaluated(self):
        path = self.docs / "samples.xlsx"
        workbook = Workbook()
        workbook.active.append(["id", "value"])
        workbook.active.append(["sample", "=1+1"])
        workbook.save(path)
        text = list(source_sections(path))[0][2]
        self.assertIn("[formula not evaluated]", text)
        self.assertNotIn("value=2", text)

    def test_chunks_preserve_original_text_and_limit(self):
        text = "RNA-seq gene_X -80C\n" * 200
        chunks = list(token_chunks(text, WordTokenizer(), 32))
        self.assertTrue(all(chunk in text for chunk in chunks))
        self.assertTrue(all(len(chunk.split()) <= 32 for chunk in chunks))

    def test_unsupported_files_are_reported(self):
        (self.docs / "reads.fastq").write_text("ACGT")
        self.assertIn("unsupported", self.ingest()["warnings"][0])

    def test_long_faq_preserves_complete_canonical_answer(self):
        from local_store import write_json
        answer = "Long instruction with an important prerequisite. " * 100
        write_json(self.docs / "faq.json", [{"question": "What is the procedure?", "answer": answer}])
        records = self.ingest()["records"]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["answer"], answer)

    def test_tsv_headers_and_sample_locations(self):
        path = self.docs / "samples.tsv"
        path.write_text("specimen\tassay\nA\tRNA-seq\n")
        record = self.ingest()["records"][0]
        self.assertIn("Columns: specimen, assay", record["text"])
        self.assertIn("sample rows 2-6", record["location"])

    def test_text_pdf_preserves_page_and_body(self):
        from pypdf import PdfWriter
        from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
        writer = PdfWriter()
        page = writer.add_blank_page(width=300, height=300)
        font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                                 NameObject("/Subtype"): NameObject("/Type1"),
                                 NameObject("/BaseFont"): NameObject("/Helvetica")})
        page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
        stream = DecodedStreamObject()
        stream.set_data(b"BT /F1 12 Tf 20 250 Td (Keep RNA frozen.) Tj ET")
        page[NameObject("/Contents")] = stream
        writer.write(self.docs / "protocol.pdf")
        record = self.ingest()["records"][0]
        self.assertIn("Keep RNA frozen.", record["text"])
        self.assertIn("page 1", record["location"])


class ClaimSupportTests(unittest.TestCase):
    def test_citation_only_is_not_an_answer(self):
        self.assertEqual(validate_grounded_answer("[S1]", 1, [{"answer": "RNA"}]),
                         (False, "empty_answer_content"))

    def test_correct_citation_with_wrong_temperature_is_rejected(self):
        sources = [{"answer": "Keep samples at -80 C."}]
        self.assertFalse(validate_grounded_answer("Keep samples at -20 C. [S1]", 1, sources)[0])

    def test_source_quote_is_supported(self):
        sources = [{"answer": "Keep samples at -80 C."}]
        self.assertTrue(validate_grounded_answer("Keep samples at -80 C. [S1]", 1, sources)[0])

    def test_quote_cannot_drop_negation(self):
        self.assertFalse(validate_grounded_answer("Use bleach. [S1]", 1, [{"answer": "Do not use bleach."}])[0])

    def test_abstention_token_does_not_bypass_validation(self):
        self.assertFalse(validate_grounded_answer("INSUFFICIENT_EVIDENCE but samples are immortal.", 1, [])[0])
