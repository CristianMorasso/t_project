#open a file and read lines
with open('sp_t0_sx_ss_1modb.las') as f:
    lines = f.readlines()
f.close()
#remove spaces from lines
lines = [line.strip() for line in lines]

with open('output.txt', 'w') as f:

    for line in lines:
        if len(line) > 0 and line[0] == "7":
            #if line contains the substring "comm", "player", "target" print it
            if "comm" in line and "player" in line and "target" in line:
                #if line contains 2 occurences of "V1" print it
                if line.count("V1") == 2 and line.count("V2") == 2 and line.count("V3") == 2:
                    # print(line)
                    f.write(line + "\n")
f.close()