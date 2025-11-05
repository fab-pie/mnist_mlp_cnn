For my hyperparameters testing, I wanted to automate testing by comparing 2 different hyperparameters together and plot an heatmap of it to have a visual representation of what is good or bad as a model.
I wanted to focus on learning rate which seams to me as one of the most interesting one to mess around, I was thinking that with a big batch, it will overfit if my learning rate is too high, or underfit if the patience is to high and the learning rate too low, so I wnated to try different combinaison to test how the accuracy, the loss and the time needed will react by changing the learning rate and an other hyperparameter. I didn't try changing STEPS parameters, I only use steps=1 to test if my code was working so that I have fast result, but for everyhting I use the standard steps=70 which seems a good compromise  on having a great accuracy and a running time not that high.

All the raw data from the hyperparameters are on the directory data plots and plots_nearest that I created

WIth all the different information from the heatmap, I already had multiple good model for my CNN as aiming more than 98% accuracy. However for the mlp I didn't go past 95% by only changing 2 parameters, but I see some paterns where accuracy decrease when patience increase, the accuracy increase when I use a large batch size but I still have to be carefull by not having too much steps or it will overfit, or that an angle of 5 is the best compromise, if it's too low it will not do anything but too high it will change the shape of the image too much.

pour MLP des supers test: JIT=1 STEPS=200 ANGLE=5 SCALE=0.2 python3 mnist_mlp.py résultat : 
lr: 2.00e-02  loss: 0.17  accuracy: 97.19%: 100%|████████████████████████████████████████████████████| 200/200 [01:39<00:00,  2.00it/s] Le résultat est excellent, et la loss est très petite.

JIT=1 STEPS=200 ANGLE=5 SCALE=0.2 LR=0.005 python3 mnist_mlp.py
lr: 5.00e-03  loss: 0.63  accuracy: 88.38%: 100%|████████████████████████████████████████████████████| 200/200 [01:39<00:00,  2.01it/s]
ram used:  0.13 GB, layers.5.bias          
exemple cas similaire mais LR différent, montre bien un détérioration du résultat car LR est plus petit, de meme si il y a learning decay qui s'active et fait baissé le learning rate, cela affecte négativement le résultat en général.

Constat pour mlp : il faut surtout pas changer le learning_rate de maniere général, ca affecte négativement les résultat et il faut avoir un learning rate 