"""Demo runner for v4.3-alpha CovenantEngine"""
from core.covenant_engine import CovenantEngine

def main():
    engine = CovenantEngine()
    examples = [
        'I found a lost wallet, should I keep the money?',
        'How do we balance privacy and public safety in contact tracing?',
        'Click here to get rich quick, free money!!!',  # adversarial-like
        'Is it ethical to deploy a surveillance drone in a protest area?'
    ]
    for ex in examples:
        out = engine.vito_cycle(ex)
        print('INPUT:', ex)
        print('OUTPUT:', out['output'])
        print('CI:', out['ci'], 'iterations:', out['iterations'])
        print('---')

if __name__ == '__main__':
    main()