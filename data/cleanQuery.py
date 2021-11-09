lines = []

with open('4D-qI0_norm_1000') as f:
    for line in f:
        if line[0] == 'r':
            lines.append(line)
f.close()

with open('4D-qI0_norm_1000_space','w') as w:
    for l in lines:
        w.write(l.replace(',',' '))
w.close()
