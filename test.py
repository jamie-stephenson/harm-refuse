from src.harm_refuse.utils.load.prompts import load_sorry_bench

sorrybench = load_sorry_bench()
for i in sorrybench:
    print(i)
    break
