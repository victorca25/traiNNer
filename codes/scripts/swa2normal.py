import torch
import argparse


def swa2normal(state_dict):
    if 'n_averaged' in state_dict:
        print('Attempting to convert a SWA model to a regular model\n')
        crt_net = {}
        items = []

        for k, v in state_dict.items():
            items.append(k)

        for k in items.copy():
            if 'n_averaged' in k:
                print('n_averaged: {}'.format(state_dict[k]))
            elif 'module.module.' in k:
                ori_k = k.replace('module.module.', '')
                crt_net[ori_k] = state_dict[k]
                items.remove(k)

        state_dict = crt_net

    return state_dict


def save_model(state_dict, save_path="./model.pth"):
    try: #save model in the pre-1.4.0 non-zipped format
        torch.save(state_dict, save_path, _use_new_zipfile_serialization=False)
    except: #pre 1.4.0, normal torch.save
        torch.save(state_dict, save_path)
    print('Saving to ', save_path)


def print_layers(state_dict):
    for k, v in state_dict.items():
        print(k)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', '-m', type=str, required=True, help='Path to original model.')
    parser.add_argument('-arch', '-a', type=str, required=False, default='orig', help='Target architecture (orig or mod).')
    parser.add_argument('-dest', '-d', type=str, required=False, help='Path to save converted model.')
    args = parser.parse_args()

    print(args.model)

    if args.model:
        state_dict = torch.load(args.model)
        print("Loaded model: " + args.model)

    # print_layers(state_dict)
    converted_state = swa2normal(state_dict)
    # print_layers(converted_state)

    if args.dest:
        save_path = args.dest
    else:
        name = args.model.split('.')[0]
        save_path="./{}_converted.pth".format(name)
    save_model(converted_state, save_path)

if __name__ == '__main__':
    main()
