from asyncore import write


with open('read.txt') as f:
    lines = f.readlines()


name = 'Md. Safirur. Rashid'

with open('write.txt', 'w') as f:
    for i,line in enumerate(lines):
        line = line[:-1]
        print(line)
        if i == 0:
            f.write('\section{Serif fonts}\n')
        elif line == 'Aboriginal Sans':
            f.write('\\newpage\n')
            f.write('\section{Sans-serif fonts}\n')
        elif line == 'AeMMono10':
            f.write('\\newpage\n')
            f.write('\section{Monospaced fonts}\n')
        elif line == 'Apropal':
            f.write('\\newpage\n')
            f.write('\section{Heavy faces}\n')
        elif line == 'Bajaderka':
            f.write('\\newpage\n')
            f.write('\section{Cursive fonts}\n')
        elif line == 'AirCut':
            f.write('\\newpage\n')
            f.write('\section{Handwriting fonts}\n')
        elif line == 'Blankenburg_UNZ1A':
            f.write('\\newpage\n')
            f.write('\section{Blackletter fonts}\n')
        elif line == 'Asana Math':
            f.write('\\newpage\n')
            f.write('\section{Math fonts}\n')
        line_n = line.replace("&","\&")
        ans = f'\setmainfont{{{line}}} {name} \qquad ({line_n}) \\\\\n'
        if i in [354,466,566,598]:
            ans = '%'+ans
        f.write(ans)