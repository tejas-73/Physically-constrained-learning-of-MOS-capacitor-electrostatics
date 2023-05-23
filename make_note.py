'''

Makes a folder for a model and saves the current running code in the folder as well as model description

'''

import os

def make_notes_and_code(model_name, model_details, code, overwrite=False, train_continue=False):
    if not overwrite and train_continue:
        os.chdir(f'./{model_name}')
        return None
    if os.path.exists(f'./{model_name}') and not overwrite:
        raise Exception(f"The file {model_name} folder already exists")
    elif not overwrite:
        os.mkdir(f'./{model_name}')
        os.chdir(f'./{model_name}')
    elif not os.path.exists(f'./{model_name}'):
        os.mkdir(f'./{model_name}')
        os.chdir(f'./{model_name}')
    else:
        os.chdir(f'./{model_name}')
    file = open(f'{model_name}.txt', 'w')
    file.write(model_details)
    file.close()
    file = open(f'{model_name}_code.py', 'w')
    file.write(code)
    file.close()