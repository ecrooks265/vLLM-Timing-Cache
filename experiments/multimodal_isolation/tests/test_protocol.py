import json
import unittest
from src.protocol import assignment, consume_sse, parse_metrics


class ProtocolTests(unittest.TestCase):
    def test_complementary_first_probes(self):
        items = [{'id': str(i)} for i in range(40)]
        a = assignment(items, 0, 0, 10)
        b = assignment(items, 0, 1, 10)
        da = {i['id']: y for i, y in a}
        db = {i['id']: y for i, y in b}
        self.assertEqual(sum(da.values()), 20)
        self.assertTrue(all(da[k] + db[k] == 1 for k in da))
        self.assertEqual(len(set(i['id'] for i, _ in a)), 40)
        self.assertNotEqual([i['id'] for i, _ in a], [i['id'] for i, _ in b])

    def test_role_is_not_first_content_and_stream_drained(self):
        consumed = []
        lines = ['data: ' + json.dumps({'choices': [{'delta': {'role': 'assistant'}}]}),
                 'data: ' + json.dumps({'choices': [{'delta': {'content': 'ok'}}]}),
                 'data: ' + json.dumps({'choices': [], 'usage': {'prompt_tokens': 8}}),
                 'data: [DONE]']
        def gen():
            for line in lines:
                consumed.append(line)
                yield line
        clock = iter([1.01, 1.03, 1.04, 1.05])
        out = consume_sse(gen(), 1, clock=lambda: next(clock))
        self.assertAlmostEqual(out['first_chunk_ms'], 10)
        self.assertAlmostEqual(out['first_content_ms'], 30)
        self.assertEqual(len(consumed), 4)
        self.assertEqual(out['usage']['prompt_tokens'], 8)

    def test_truncated_stream_rejected(self):
        with self.assertRaises(RuntimeError):
            consume_sse(['data: {"choices":[{"delta":{"content":"x"}}]}'], 0)

    def test_metrics_missing_is_not_zero(self):
        self.assertIsNone(parse_metrics('')['prefix_cache_hits'])
        out = parse_metrics('vllm:prefix_cache_hits_total{engine="0"} 16\n'
                            'vllm:prefix_cache_hits_total{engine="1"} 32\n'
                            'vllm:prefix_cache_queries_total 64\n'
                            'vllm:gpu_prefix_cache_hits_total 999\n')
        self.assertEqual(out['prefix_cache_hits'], 48)
        self.assertEqual(out['prefix_cache_queries'], 64)


if __name__ == '__main__':
    unittest.main()
