import os

if __name__ == '__main__':

    count = 0
    patterns = {"0":0,"1":0, "2":0, "3": 0, "4":0, "5":0, "6":0, "7":0, "8":0}
    with open(os.path.join(os.pardir,os.pardir, "outnew", "pattern_v4", "output_21885.txt"), "r") as ins:
        for line in ins:
            if "Pattern(s):" in line:
                pattern = line.split()
                count += 1
                for i in patterns.keys():
                    if i in pattern:
                        patterns[i] += 1

    print(count)
    print(patterns)
