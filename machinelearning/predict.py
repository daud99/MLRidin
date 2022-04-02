

def predict(output_file, buffer_size=10):
    line_count = 0
    lines = []
    with open(output_file) as csv_file:
        while(True):
            line = csv_file.readline()
            if not line:
                continue
            lines.append(line)
            line_count += 1
            if(line_count == buffer_size):
                print('################### START #############')
                print(lines)
                print('################### END #############')
                line_count = 0
                lines = []
