import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import os


def main():
    work_dir = Path(__file__).parent.absolute()
    reward_files_path = work_dir / 'optRLFiles'


    for file in reward_files_path.iterdir():
        file_name = file.name
        if file_name.startswith('rs_'):
            new_name = 'rs_' + file_name[3:].zfill(4)
            os.rename(reward_files_path / file_name, reward_files_path / new_name)
            

    tot_rewards = []

    for file in reward_files_path.iterdir():
        if file.name.startswith('rs_'):
            filehandler =  open(reward_files_path / file.name, 'rb') 
            tmp_rs = pickle.load(filehandler)
            tmp_tot_rs = 0
            for el in tmp_rs:
                tmp_tot_rs += sum(el)
            tot_rewards.append(tmp_tot_rs)
            print("{} - {}".format(file.name, tmp_tot_rs))

    plt.plot(tot_rewards)
    plt.show()



if __name__ == '__main__':
    main()