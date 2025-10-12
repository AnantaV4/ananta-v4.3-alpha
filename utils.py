"""Utilities: simple ledger persistence for alpha demo"""
import os, json
LEDGER_PATH = '/mnt/data/ananta_v4_3_alpha_demo_ledger.jsonl'

def append_ledger(record):
    try:
        with open(LEDGER_PATH, 'a') as f:
            f.write(json.dumps(record) + '\n')
    except Exception as e:
        print('Failed to append ledger:', e)