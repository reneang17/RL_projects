#!/bin/bash

conda create --name drlnd python=3.6; \
conday activate drlnd; \

git clone https://github.com/openai/gym; \

#Install openai utilities
cd gym; \
pip install -e .; \
pip install -e .'[box2d]'; \
pip install -e .'[classic_control]'; \
cd ..; \

#Install necessary python
git clone https://github.com/udacity/deep-reinforcement-learning; \
cd deep-reinforcement-learning/python; \
pip install .; \
cd ../../; \

#set up python kernel for drlnd env
python -m ipykernel install --user --name drlnd --display-name "drlnd"; \

#clone my github project
git clone https://github.com/reneang17/RL_projects; \
cd RL_projects/collector; \

#dowload and unzip Udacity visual enviroment
curl https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip --output Banana.app.zip; \
unzip Banana.app.zip; \
