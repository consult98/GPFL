
#mnist
    
    #FL
    python main.py -device gpu -W 200 -C 1 -T 200 -E 1 -B 20 -lr 0.1 -dset mnist -alg fedavg -bn_private none

    # MTFL
    python main.py -device gpu -W 200 -C 1 -T 200 -E 1 -B 20 -lr 0.1 -dset mnist -alg MTFL -bn_private yb
    
    # PFedSA
    python main.py -device gpu -W 200 -C 1 -T 200 -E 1 -B 20 -lr 0.1 -dset mnist -alg PFedSA -bn_private yb -preT 1 -distance 0.6
    
    # PFedMe
    python main.py -device gpu -W 200 -C 1 -T 200 -E 1 -B 20 -lr 0.3 -dset mnist -alg pfedme -lamda 1 -beta 1

    #SFL
    python main.py -device gpu -W 200 -C 1 -T 200 -E 1 -B 20 -lr 0.1 -dset mnist -alg SFL

    #GPFL
    python main.py -device gpu -W 200 -C 1 -T 200 -E 1 -B 20 -lr 0.1 -dset mnist -alg GPFL -bn_private yb -preT 1 -distance 0.6
    #GPFL(a=0.5)
    python main.py -device gpu -W 200 -C 1 -T 200 -E 1 -B 20 -lr 0.1 -dset mnist -alg GPFL -bn_private yb -preT 1 -distance 0.6 -adjbeta 0.5
    #GPFL(a=0.8)
    python main.py -device gpu -W 200 -C 1 -T 200 -E 1 -B 20 -lr 0.1 -dset mnist -alg GPFL -bn_private yb -preT 1 -distance 0.6 -adjbeta 0.8


#cifar

    #FL
    python main.py -device gpu -W 200 -C 1 -T 200 -E 1 -B 20 -lr 0.1 -dset cifar10 -alg fedavg -bn_private none
    
    # MTFL
    python main.py -device gpu -W 200 -C 1 -T 200 -E 1 -B 20 -lr 0.1 -dset cifar10 -alg MTFL -bn_private yb

    # PFedSA
    python main.py -device gpu -W 200 -C 1 -T 200 -E 1 -B 20 -lr 0.1 -dset cifar10 -alg PFedSA -bn_private yb -preT 3 -distance 0.8

    # PFedMe
    python main.py -device gpu -W 200 -C 1 -T 200 -E 1 -B 20 -lr 0.3 -dset cifar10 -alg pfedme -lamda 1 -beta 1

    #SFL
    python main.py -device gpu -W 200 -C 1 -T 200 -E 1 -B 20 -lr 0.1 -dset cifar10 -alg SFL

    #GPFL
    python main.py -device gpu -W 200 -C 1 -T 200 -E 1 -B 20 -lr 0.1 -dset cifar10 -alg GPFL -bn_private yb -preT 3 -distance 0.8
    #GPFL(a=0.5)
    python main.py -device gpu -W 200 -C 1 -T 200 -E 1 -B 20 -lr 0.1 -dset cifar10 -alg GPFL -bn_private yb -preT 3 -distance 0.8 -adjbeta 0.5
    #GPFL(a=0.8)
    python main.py -device gpu -W 200 -C 1 -T 200 -E 1 -B 20 -lr 0.1 -dset cifar10 -alg GPFL -bn_private yb -preT 3 -distance 0.8 -adjbeta 0.8

#metr-la

    #FL
    python main.py -device gpu -W 207 -C 1 -T 50 -E 1 -B 128 -lr 0.01 -dset metr-la -alg fedavg -bn_private none
    
    # MTFL
    python main.py -device gpu -W 207 -C 1 -T 50 -E 1 -B 128 -lr 0.01 -dset metr-la -alg MTFL -bn_private yb
    
    #SFL
    python main.py -device gpu -W 207 -C 1 -T 50 -E 1 -B 128 -lr 0.01 -dset metr-la -alg SFL

    #GPFL
    python main.py -device gpu -W 207 -C 1 -T 50 -E 1 -B 128 -lr 0.01 -dset metr-la -alg GPFL -bn_private yb -preT 3 -distance 0.2 -adjbeta 0.2

#pems-bay
     #FL
    python main.py -device gpu -W 325 -C 1 -T 20 -E 1 -B 128 -lr 0.01 -dset pems-bay -alg fedavg -bn_private none
    
    # MTFL
    python main.py -device gpu -W 325 -C 1 -T 20 -E 1 -B 128 -lr 0.01 -dset pems-bay -alg MTFL -bn_private yb

    #SFL
    python main.py -device gpu -W 325 -C 1 -T 20 -E 1 -B 128 -lr 0.01 -dset pems-bay -alg SFL

    #GPFL
    python main.py -device gpu -W 325 -C 1 -T 20 -E 1 -B 128 -lr 0.01 -dset pems-bay -alg GPFL -bn_private yb -preT 3 -distance 0.2 -adjbeta 0.8


