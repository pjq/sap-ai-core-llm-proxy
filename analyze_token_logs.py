#!/usr/bin/env python3
"""
Token Usage Log Analyzer

Analyzes logs/token_usage.log to provide insights into API usage,
token consumption, and costs.
"""

import json
import sys
from collections import defaultdict
from datetime import datetime
import argparse

def parse_json_log(line):
    """Parse a JSON log line. Returns None if not JSON format."""
    try:
        # Extract JSON part from log line (after timestamp)
        if ' - {' in line:
            json_part = line.split(' - ', 1)[1]
            return json.loads(json_part)
        return None
    except (json.JSONDecodeError, IndexError):
        return None

def parse_legacy_log(line):
    """Parse plain text format log line."""
    import re

    data = {}

    # Extract timestamp
    timestamp_match = re.match(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
    if timestamp_match:
        data['timestamp'] = timestamp_match.group(1)

    # Extract fields
    patterns = {
        'user': r'User: ([^,]+)',
        'ip': r'IP: ([^,]+)',
        'model': r'Model: ([^,]+)',
        'subaccount': r'SubAccount: ([^,]+)',
        'prompt': r'PromptTokens: (\d+)',
        'completion': r'CompletionTokens: (\d+)',
        'total': r'TotalTokens: (\d+)',
        'duration': r'Duration: (\d+)ms'
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, line)
        if match:
            value = match.group(1).strip()
            if key in ['prompt', 'completion', 'total', 'duration']:
                data[key] = int(value)
            else:
                data[key] = value

    data['streaming'] = '(Streaming)' in line

    return data if data else None

def analyze_logs(log_file, start_date=None, end_date=None, model_filter=None, subaccount_filter=None):
    """Analyze token usage logs."""

    stats = {
        'total_calls': 0,
        'streaming_calls': 0,
        'non_streaming_calls': 0,
        'successful_calls': 0,
        'error_calls': 0,
        'streaming_zero_tokens': 0,
        'total_tokens': 0,
        'total_duration_ms': 0,
        'by_model': defaultdict(lambda: {'calls': 0, 'tokens': 0, 'duration_ms': 0}),
        'by_subaccount': defaultdict(lambda: {'calls': 0, 'tokens': 0, 'duration_ms': 0}),
        'by_user': defaultdict(lambda: {'calls': 0, 'tokens': 0, 'duration_ms': 0}),
        'by_date': defaultdict(lambda: {'calls': 0, 'tokens': 0}),
        'by_hour': defaultdict(lambda: {'calls': 0, 'tokens': 0}),
    }

    with open(log_file, 'r') as f:
        for line in f:
            # Try parsing as JSON first (new format)
            entry = parse_json_log(line)

            # Fall back to legacy format
            if not entry:
                entry = parse_legacy_log(line)

            if not entry:
                continue

            # Apply filters
            if model_filter and entry.get('model') != model_filter:
                continue
            if subaccount_filter and entry.get('subaccount') != subaccount_filter:
                continue

            # Extract data (handle both JSON and plain text formats)
            if 'tokens' in entry:  # JSON format (if used)
                model = entry.get('model', 'unknown')
                subaccount = entry.get('subaccount', 'unknown')
                user = entry.get('user', 'unknown')
                total_tokens = entry['tokens'].get('total', 0)
                prompt_tokens = entry['tokens'].get('prompt', 0)
                completion_tokens = entry['tokens'].get('completion', 0)
                is_streaming = entry.get('streaming', False)
                duration_ms = entry.get('duration_ms', 0)
                status = entry.get('status', 'success')
                timestamp = entry.get('timestamp', '')
            else:  # Plain text format
                model = entry.get('model', 'unknown')
                subaccount = entry.get('subaccount', 'unknown')
                user = entry.get('user', 'unknown')
                total_tokens = entry.get('total', 0)
                prompt_tokens = entry.get('prompt', 0)
                completion_tokens = entry.get('completion', 0)
                is_streaming = entry.get('streaming', False)
                duration_ms = entry.get('duration', 0)  # Duration field in plain text
                status = 'success'
                timestamp = entry.get('timestamp', '')

            # Update statistics
            stats['total_calls'] += 1

            if is_streaming:
                stats['streaming_calls'] += 1
                if total_tokens == 0:
                    stats['streaming_zero_tokens'] += 1
            else:
                stats['non_streaming_calls'] += 1

            if status == 'success':
                stats['successful_calls'] += 1
            else:
                stats['error_calls'] += 1

            stats['total_tokens'] += total_tokens
            stats['total_duration_ms'] += duration_ms

            # By model
            stats['by_model'][model]['calls'] += 1
            stats['by_model'][model]['tokens'] += total_tokens
            stats['by_model'][model]['duration_ms'] += duration_ms

            # By subaccount
            stats['by_subaccount'][subaccount]['calls'] += 1
            stats['by_subaccount'][subaccount]['tokens'] += total_tokens
            stats['by_subaccount'][subaccount]['duration_ms'] += duration_ms

            # By user
            stats['by_user'][user]['calls'] += 1
            stats['by_user'][user]['tokens'] += total_tokens
            stats['by_user'][user]['duration_ms'] += duration_ms

            # By date and hour
            if timestamp:
                try:
                    if 'T' in timestamp:  # ISO format
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    else:  # Legacy format
                        dt = datetime.strptime(timestamp[:19], '%Y-%m-%d %H:%M:%S')

                    date_key = dt.strftime('%Y-%m-%d')
                    hour_key = dt.strftime('%Y-%m-%d %H:00')

                    stats['by_date'][date_key]['calls'] += 1
                    stats['by_date'][date_key]['tokens'] += total_tokens

                    stats['by_hour'][hour_key]['calls'] += 1
                    stats['by_hour'][hour_key]['tokens'] += total_tokens
                except:
                    pass

    return stats

def print_report(stats):
    """Print formatted analysis report."""

    print("=" * 80)
    print("TOKEN USAGE ANALYSIS REPORT")
    print("=" * 80)

    print(f"\n📊 OVERALL STATISTICS")
    print(f"{'Total API Calls:':<30} {stats['total_calls']:>10,}")
    print(f"  {'Streaming calls:':<28} {stats['streaming_calls']:>10,}")
    print(f"  {'Non-streaming calls:':<28} {stats['non_streaming_calls']:>10,}")
    print(f"{'Successful calls:':<30} {stats['successful_calls']:>10,}")
    print(f"{'Error calls:':<30} {stats['error_calls']:>10,}")
    print(f"{'Streaming with 0 tokens:':<30} {stats['streaming_zero_tokens']:>10,}")

    print(f"\n💰 TOKEN USAGE")
    print(f"{'Total tokens consumed:':<30} {stats['total_tokens']:>10,}")
    if stats['total_calls'] > 0:
        avg_tokens = stats['total_tokens'] / stats['total_calls']
        print(f"{'Average tokens per call:':<30} {avg_tokens:>10,.1f}")

    print(f"\n⏱️  PERFORMANCE")
    if stats['total_duration_ms'] > 0:
        avg_duration = stats['total_duration_ms'] / stats['total_calls']
        print(f"{'Average request duration:':<30} {avg_duration:>10,.0f} ms")

    # By model
    print(f"\n📱 BY MODEL (Top 10)")
    print(f"{'Model':<30} {'Calls':>8} {'Tokens':>12} {'Avg/Call':>10}")
    print("-" * 80)
    for model, data in sorted(stats['by_model'].items(), key=lambda x: x[1]['calls'], reverse=True)[:10]:
        avg = data['tokens'] / data['calls'] if data['calls'] > 0 else 0
        print(f"{model:<30} {data['calls']:>8,} {data['tokens']:>12,} {avg:>10,.0f}")

    # By subaccount
    print(f"\n🏢 BY SUBACCOUNT")
    print(f"{'SubAccount':<30} {'Calls':>8} {'Tokens':>12} {'Avg/Call':>10}")
    print("-" * 80)
    for subaccount, data in sorted(stats['by_subaccount'].items(), key=lambda x: x[1]['calls'], reverse=True):
        avg = data['tokens'] / data['calls'] if data['calls'] > 0 else 0
        print(f"{subaccount:<30} {data['calls']:>8,} {data['tokens']:>12,} {avg:>10,.0f}")

    # By user (top 10)
    print(f"\n👤 BY USER (Top 10)")
    print(f"{'User':<35} {'Calls':>8} {'Tokens':>12} {'Avg/Call':>10}")
    print("-" * 80)
    for user, data in sorted(stats['by_user'].items(), key=lambda x: x[1]['calls'], reverse=True)[:10]:
        avg = data['tokens'] / data['calls'] if data['calls'] > 0 else 0
        print(f"{user:<35} {data['calls']:>8,} {data['tokens']:>12,} {avg:>10,.0f}")

    # By date (last 7 days)
    print(f"\n📅 BY DATE (Last 7 days)")
    print(f"{'Date':<20} {'Calls':>8} {'Tokens':>12} {'Avg/Call':>10}")
    print("-" * 80)
    for date, data in sorted(stats['by_date'].items(), reverse=True)[:7]:
        avg = data['tokens'] / data['calls'] if data['calls'] > 0 else 0
        print(f"{date:<20} {data['calls']:>8,} {data['tokens']:>12,} {avg:>10,.0f}")

    print("\n" + "=" * 80)

def main():
    parser = argparse.ArgumentParser(description='Analyze token usage logs')
    parser.add_argument('--log-file', default='logs/token_usage.log',
                       help='Path to token usage log file (default: logs/token_usage.log)')
    parser.add_argument('--model', help='Filter by model name')
    parser.add_argument('--subaccount', help='Filter by subaccount name')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')

    args = parser.parse_args()

    try:
        stats = analyze_logs(
            args.log_file,
            start_date=args.start_date,
            end_date=args.end_date,
            model_filter=args.model,
            subaccount_filter=args.subaccount
        )
        print_report(stats)
    except FileNotFoundError:
        print(f"Error: Log file not found: {args.log_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error analyzing logs: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
