import json
import tempfile
import unittest
from pathlib import Path

import torch

from exp43_semantic_corpus_evaluation import load_requests, summarize_requests
from moran_semantic_isp import AXES


class SemanticCorpusEvaluationTests(unittest.TestCase):
    def test_loads_word_phrase_robustness_and_filter_groups(self):
        corpus = {
            "anchors": ["vivid"],
            "seen_words": ["warm"],
            "unseen_test_words": ["angelic"],
            "compound_phrases": ["warm sunset"],
            "robustness_test": {"misspelled": ["dremy"], "slang": ["vibes"],
                                "multilingual": ["luminoso"]},
            "instagram_filters": {"filters": ["Clarendon"]},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.json"
            path.write_text(json.dumps(corpus))
            requests = load_requests(path)
        self.assertEqual(len(requests), 8)
        self.assertIn("a warm sunset photo", [request["clip_prompt"] for request in requests])
        self.assertIn("a photo processed with the Clarendon filter",
                      [request["clip_prompt"] for request in requests])
        self.assertEqual(requests[-1]["category"], "named_filter")

    def test_coordinates_are_summarized_for_matching_request(self):
        requests = [{"request_id": "a", "text": "warm", "category": "word"},
                    {"request_id": "b", "text": "dreamy", "category": "word"}]
        coordinates = torch.tensor([[.1, .2, .3, .4, .5], [-.1, -.2, -.3, -.4, -.5]])
        rows = [
            {"request_id": "a", "cosine_gain": .01},
            {"request_id": "a", "cosine_gain": -.02},
            {"request_id": "b", "cosine_gain": .03},
        ]
        result = summarize_requests(requests, coordinates, rows)
        for axis, expected in zip(AXES, [.1, .2, .3, .4, .5]):
            self.assertAlmostEqual(result[0][axis], expected, places=6)
        for axis, expected in zip(AXES, [-.1, -.2, -.3, -.4, -.5]):
            self.assertAlmostEqual(result[1][axis], expected, places=6)
        self.assertAlmostEqual(result[0]["mean_cosine_gain"], -.005)
        self.assertEqual(result[0]["positive_gain_images"], 1)


if __name__ == "__main__":
    unittest.main()