working command:

nn on linear

 python test_fairnn.py --mode 2 --n 10000 --seed 0 --niters 50000 --final_temp 0.1 --diter 10


python test_fairnn.py --mode 8 --n 10000 --niters 75000 --final_temp 0.05 --diter 3 --giter 1 --batch_size 64 --offset -3 --seed 9

python test_fairll_guni.py --mode 3 --n 10000 --niters 50000 --final_temp 0.05 --diter 3 --giter 1 --batch_size 64 --offset -3 --seed 9

python test_fairll_guni.py --mode 3 --n 10000 --niters 50000 --final_temp 0.05 --diter 3 --batch_size 64 --seed 1 --dim_x 70