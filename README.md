# footballer-price
Use neural networks to calculate the price of a football (soccer) player by using data from FIFA 2017.

MAIN FILE: *nn.py*, uses Python,Theano, Keras. Can either run a new network, or load a pre-saved Keras model.

*data_files*: manually prepared txt files containing HTML source code for FIFA 2017 player stats or features. Goalkeepers are excluded. Data extracted using *make_dataset.py*

<http://sofifa.com/players?gender=0&pn%5B0%5D=27&pn%5B1%5D=25&pn%5B2%5D=23&pn%5B3%5D=22&pn%5B4%5D=21&pn%5B5%5D=20&pn%5B6%5D=18&pn%5B7%5D=16&pn%5B8%5D=14&pn%5B9%5D=12&pn%5B10%5D=10&pn%5B11%5D=8&pn%5B12%5D=7&pn%5B13%5D=5&pn%5B14%5D=3&pn%5B15%5D=2&col=vl&sort=desc&offset=0>

To get following pages, add 100 to offset

Features are arranged as:
	
	Acceleration, Aggression, Agility, Balance, Ball Control, Composure, Crossing, Curve, DEF, DRI,
    Dribbling, Finishing, Free Kick Accuracy, Heading Accuracy, Interceptions, Jumping, Long Passing,
    Long Shots, Marking, OVA, PAC, PAS, Penalties, PHY, Positioning, POT, Reactions, SHO, Short Passing,
    Shot Power, Sliding Tackle, Sprint Speed, Stamina, Standing Tackle, Strength, Vision, Volleys
Above 37 are all out of 100. Next is age, and last 3 are stars out of 5
    
	Age, International Reputation, Skill Moves, Weak Foot
Finally is Value, i.e. player price, which is the output variable

Sourya's account credentials on sofifa automatically arranges stats like this. For others, you need to do it once manually, then login to the website to save this layout.

Prices are quantized and made one-hot. Features are normalized. Data is shuffled and split into training and test, then fed to neural network.

Models optimized via Keras inbuilt features, best models used to compute accuracy, accuracy results stored in *final_storevar.txt*

*model_files*: Best Keras models


