# Modeling-crypto-prices
Code for the project "Modeling stock prices with Brownian motion".

### The structure of the root directory is:

    .
    ├── classes                   
        ├── __init__.py 
        ├──cryptocurrency.py
        ├──GBM_base.py
    ├──data
        ├──(Crypto-ticker)_USD.csv
    ├──results
    ├──main.py
    ├──simulation.py
    ├──requirements.txt
    ├──Literature Review

There are two files that may be run by a user - main.py and simulation.py.
1. The main.py file is used to generate GBM models for cryptocurrency price prediction and visualize the results. The script is designed to accept arguments, which are described in detail within the file. 
2. The simulation.py file is used to run the simulator. Instructions for the arguments may be found in the file.
