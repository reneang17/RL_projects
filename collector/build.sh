#!/bin/bash


#Install openai utilities
git clone https://github.com/openai/gym; \
pip install -e ./gym; \
pip install -e ./gym'[box2d]'; \
pip install -e ./gym'[classic_control]'; \
rm -rf gym

#Install necessary python
git clone https://github.com/udacity/deep-reinforcement-learning; \
pip install ./deep-reinforcement-learning/python/; \
rm -rf deep-reinforcement-learning

#set up python kernel for drlnd env
python -m ipykernel install --user --name drlnd --display-name "drlnd"; \

# remove unnecessary clones
rm -rf gym deep-reinforcement-learning

#clone my github project
cd collector; \


#dowload and unzip Udacity visual enviroment
curl https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip --output Banana.app.zip; \
unzip Banana.app.zip;
