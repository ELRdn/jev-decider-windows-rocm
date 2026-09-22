import asyncio
import copy
import importlib.util
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from service import local_rocm_server as server
BODY = {'state': 'test', 'questions': {'x': {'type': 'noul', 'instructions': 'test?'}}}


class LowMemoryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.memory_patch = patch.object(server, 'gpu_memory', return_value={})
        self.memory_patch.start()

    async def asyncTearDown(self):
        self.memory_patch.stop()

    async def client(self):
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url='http://test')

    async def ready(self, client):
        for _ in range(100):
            if (await client.get('/health')).json()['ready']:
                return
            await asyncio.sleep(.01)
        self.fail('Backend did not initialize')

    async def test_startup_is_cpu_only_and_first_request_wakes_gpu(self):
        model = SimpleNamespace(dev='cpu')
        def inference(runtime, req):
            self.assertIs(runtime.model, model)
            runtime.model.dev = 'cuda'
            runtime.phase = 'ready'
            return {'answers': {'x': {'type': 'noul', 'noul': .2}}}
        with patch.object(server, 'load_model', return_value=model), patch.object(server, 'execute_request', side_effect=inference) as execute:
            async with server.lifespan(server.app):
                async with await self.client() as client:
                    await self.ready(client)
                    status = (await client.get('/health')).json()
                    self.assertEqual(status['phase'], 'idle')
                    self.assertFalse(status['gpu_resident'])
                    execute.assert_not_called()
                    self.assertEqual((await client.post('/v1/systemone', json=BODY)).status_code, 200)
                    self.assertTrue((await client.get('/health')).json()['gpu_resident'])

    async def test_idle_offload_rejects_racing_request_and_can_resume(self):
        entered, release = threading.Event(), threading.Event()
        model = SimpleNamespace(dev='cpu')
        def inference(runtime, req):
            runtime.model.dev = 'cuda'
            runtime.phase = 'ready'
            return {'answers': {}}
        def offload(runtime):
            entered.set()
            release.wait(3)
            runtime.model.dev = 'cpu'
        with patch.object(server, 'IDLE_SECONDS', .01), patch.object(server, 'load_model', return_value=model), \
             patch.object(server, 'execute_request', side_effect=inference), patch.object(server, 'release_gpu', side_effect=offload):
            async with server.lifespan(server.app):
                async with await self.client() as client:
                    await self.ready(client)
                    self.assertEqual((await client.post('/v1/systemone', json=BODY)).status_code, 200)
                    self.assertTrue(await asyncio.to_thread(entered.wait, 2))
                    self.assertEqual((await client.post('/v1/systemone', json=BODY)).status_code, 503)
                    release.set()
                    for _ in range(100):
                        if (await client.get('/health')).json()['phase'] == 'idle':
                            break
                        await asyncio.sleep(.01)
                    self.assertFalse((await client.get('/health')).json()['gpu_resident'])
                    self.assertEqual((await client.post('/v1/systemone', json=BODY)).status_code, 200)

    async def test_disconnect_does_not_allow_parallel_gpu_work(self):
        entered, release = threading.Event(), threading.Event()
        def inference(runtime, req):
            runtime.phase = 'ready'
            entered.set()
            release.wait(3)
            return {'answers': {}}
        with patch.object(server, 'load_model', return_value=SimpleNamespace(dev='cpu')), \
             patch.object(server, 'execute_request', side_effect=inference):
            async with server.lifespan(server.app):
                async with await self.client() as client:
                    await self.ready(client)
                    request = asyncio.create_task(client.post('/v1/systemone', json=BODY))
                    self.assertTrue(await asyncio.to_thread(entered.wait, 2))
                    request.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await request
                    self.assertTrue((await client.get('/health')).json()['busy'])
                    self.assertEqual((await client.post('/v1/systemone', json=BODY)).status_code, 503)
                    release.set()
                    for _ in range(100):
                        if not (await client.get('/health')).json()['busy']:
                            break
                        await asyncio.sleep(.01)
                    self.assertEqual((await client.post('/v1/systemone', json=BODY)).status_code, 200)

    async def test_loading_and_failure_are_not_ready(self):
        release = threading.Event()
        def fail(_):
            release.wait(3)
            raise RuntimeError('simulated load failure')
        with patch.object(server, 'load_model', side_effect=fail):
            async with server.lifespan(server.app):
                async with await self.client() as client:
                    self.assertFalse((await client.get('/health')).json()['ready'])
                    self.assertEqual((await client.post('/v1/systemone', json=BODY)).status_code, 503)
                    release.set()
                    for _ in range(100):
                        if (await client.get('/health')).json()['phase'] == 'failed':
                            break
                        await asyncio.sleep(.01)
                    self.assertEqual((await client.get('/health')).json()['phase'], 'failed')
                    self.assertFalse((await client.get('/health')).json()['ready'])

    async def test_deep_idle_recycles_used_gpu_but_keeps_cpu_only_startup(self):
        with patch.object(server, 'DEEP_IDLE_SECONDS', .01), patch.object(server, 'load_model', return_value=SimpleNamespace(dev='cpu')):
            async with server.lifespan(server.app):
                async with await self.client() as client:
                    await self.ready(client)
                    await asyncio.sleep(.55)
                    self.assertEqual((await client.get('/health')).json()['phase'], 'idle')
                    server.app.state.runtime.stats['resumes'] = 1
                    await asyncio.sleep(.55)
                    status = (await client.get('/health')).json()
                    self.assertEqual(status['phase'], 'recycle_pending')
                    self.assertFalse(status['ready'])
                    self.assertEqual((await client.post('/v1/systemone', json=BODY)).status_code, 503)

    def test_invalid_request_does_not_upload_model(self):
        model = SimpleNamespace(m=SimpleNamespace(to=lambda _: self.fail('GPU uploaded')))
        runtime = SimpleNamespace(model=model)
        with patch.object(server, 'validate_request', side_effect=server.HTTPException(413, 'too large')):
            with self.assertRaises(server.HTTPException) as exc:
                server.execute_request(runtime, server.SystemOneRequest(**BODY))
            self.assertEqual(exc.exception.status_code, 413)

    def test_questions_and_model_probabilities_are_not_rewritten(self):
        original = {'tool': {'type': 'choice', 'criteria': {'read_file': 'Read file', 'no_tool_needed': 'Reply'}},
                    'needs_tool': {'type': 'noul', 'instructions': 'Need a tool?'}}
        before = copy.deepcopy(original)
        prepared = server.prepare_questions(original)
        self.assertEqual(original, before)
        self.assertEqual(prepared['tool'], original['tool'])
        self.assertIn('read_file', prepared['needs_tool']['instructions'])
        with self.assertRaises(RuntimeError):
            server.check_finite({'answers': {'x': {'noul': float('nan')}}})

    @unittest.skipUnless(importlib.util.find_spec("decider"), "Optional Decider dependency is not installed")
    def test_padding_preserves_tokens_and_answer_positions(self):
        import decider.infer as infer
        original = infer.collate
        try:
            server.install_shape_buckets()
            items = [{'ids': list(range(130)), 'slots': [129], 'golds': [0], 'nopts': [2]}]
            batch = infer.collate(items, 0)
            self.assertEqual(tuple(batch['input_ids'].shape), (1, 256))
            self.assertEqual(batch['input_ids'][0, :130].tolist(), list(range(130)))
            self.assertEqual(int(batch['attention_mask'].sum()), 130)
            self.assertEqual(batch['slot_idx'].tolist(), [129])
        finally:
            infer.collate = original


if __name__ == '__main__':
    unittest.main()
