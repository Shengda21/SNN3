"""R3 entry point using the retained R2.1 numerical training implementation."""
import argparse
from pathlib import Path
import r2_vision as base

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--T',type=int,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--lr',type=float,required=True)
    p.add_argument('--steps',type=int,default=2048)
    p.add_argument('--tag',required=True)
    p.add_argument('--stage',required=True)
    p.add_argument('--microbatch',type=int,default=64)
    p.add_argument('--workers',type=int,default=6)
    p.add_argument('--initial',default=str(base.INIT))
    a=p.parse_args()
    base.INIT=Path(a.initial)
    base.train(a)
