import os
import json

def extract_from_file(file_path):
    dialog = []
    with open(file_path) as f:
        loaded_json = json.load(f)
    turns = loaded_json['turns']
    input_text = ''
    output_text = ''
    for turn in turns:
        if turn['speaker'] == 'U':
            input_text = turn['utterance']
        else:
            output_text = turn['utterance']
            breakdown_count = 0
            ungrammatical_count = 0
            for annotation in turn['annotations']:
                if annotation['breakdown'] == 'T':
                    breakdown_count += 0.5
                elif annotation['breakdown'] == 'X':
                    breakdown_count += 1
                if annotation['ungrammatical-sentence'] == 'X':
                    ungrammatical_count
            breakdown_count /= len(turn['annotations'])
            ungrammatical_count /= len(turn['annotations'])
            if breakdown_count <= 0.5 and ungrammatical_count <= 0.5:
                if input_text != '':
                    dialog.append((input_text, output_text))
            input_text = ''
    return dialog

def main():
    dialog = []
    for file in os.listdir('json/init100/'):
        dialog += extract_from_file(f'json/init100/{file}')
    for file in os.listdir('json/rest1046/'):
        dialog += extract_from_file(f'json/rest1046/{file}')
    with open('dialog.txt', 'w') as f:
        for element in dialog:
            f.write(f'input: {element[0]}\n')
            f.write(f'output: {element[1]}\n')

if __name__ == '__main__':
    main()
