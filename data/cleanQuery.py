lines = []

with open('8D-qI0_norm') as f:
    for line in f:
        if line[0] == 'r':
            lines.append(line)
f.close()

with open('8D-qI0_norm-clean','w') as w:
    for l in lines:
        w.write(l)
w.close()
