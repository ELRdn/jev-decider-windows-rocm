"""Synthetic local inference probe. No selected tool is ever executed."""
import argparse
import json
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--rounds', type=int, default=3)
    args = parser.parse_args()
    url = args.base_url.rstrip('/')
    if urllib.parse.urlparse(url).hostname not in ('localhost', '127.0.0.1', '::1'):
        parser.error('This probe is for a local service only.')
    if not 1 <= args.rounds <= 100:
        parser.error('--rounds must be between 1 and 100')
    body = {'state': 'My card was charged twice and I want a refund.', 'questions': {
        'department': {'type': 'choice', 'instructions': 'Which team should handle this?',
                       'criteria': {'billing': 'charges invoices refunds', 'technical': 'bugs outages', 'sales': 'purchases'}}}}
    results = []
    for _ in range(args.rounds):
        request = urllib.request.Request(url + '/v1/systemone', json.dumps(body).encode(), {'Content-Type': 'application/json'})
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = json.load(response)
        except urllib.error.HTTPError as error:
            raise SystemExit(f'HTTP {error.code}: {error.read(300).decode(errors="replace")}') from error
        assert data['model'] == 'decider-v10', data['model']
        assert data['answers']['department']['choice'] == 'billing', data['answers']
        results.append({'ms': round((time.perf_counter()-started)*1000, 2), 'result': data})
    with urllib.request.urlopen(url + '/health', timeout=5) as response:
        status = json.load(response)
    print(json.dumps({'calls': results, 'median_ms': statistics.median(x['ms'] for x in results),
                      'phase': status['phase'], 'hardware': status['hardware']}, indent=2))


if __name__ == '__main__':
    main()
