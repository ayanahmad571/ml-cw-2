# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# This template was originally adapted to KCL by Simon Parsons, but then
# revised and updated to Py3 for the 2022 course by Dylan Cope and Lin Li

from __future__ import absolute_import
from __future__ import print_function

import random
import sys

from pacman import Directions, GameState
from pacman_utils.game import Agent
from pacman_utils import util


class GameStateFeatures:
    """
    Wrapper class around a game state where you can extract
    useful information for your Q-learning algorithm

    WARNING: We will use this class to test your code, but the functionality
    of this class will not be tested itself
    """

    def __init__(self, state: GameState):
        """
        Args:
            state: A given game state object
        """

        
        self.legal = state.getLegalPacmanActions()
        self.packmanPosition = state.getPacmanPosition()
        self.ghostPositions = state.getGhostPositions()
        self.capsules = state.getCapsules()
        self.food = state.getFood()
        self.walls = state.getWalls()
        if Directions.STOP in self.legal:
            self.legal.remove(Directions.STOP)
        
    def __str__(self) -> str:
        stringHash = str(self.legal) + str(self.packmanPosition) +str(self.ghostPositions)+str(self.capsules)+str(self.food) + str(self.walls)
        return stringHash
    


class QLearnAgent(Agent):

    def __init__(self,
                 alpha: float = 0.2,
                 epsilon: float = 0.05,
                 gamma: float = 0.8,
                 maxAttempts: int = 30,
                 numTraining: int = 10):
        """
        These values are either passed from the command line (using -a alpha=0.5,...)
        or are set to the default values above.

        The given hyperparameters are suggestions and are not necessarily optimal
        so feel free to experiment with them.

        Args:
            alpha: learning rate
            epsilon: exploration rate
            gamma: discount factor
            maxAttempts: How many times to try each action in each state
            numTraining: number of training episodes
        """
        super().__init__()
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.maxAttempts = int(maxAttempts)
        self.numTraining = int(numTraining)

        # Count the number of games we have played
        self.episodesSoFar = 0

        self.qTableValues = util.Counter() # Storing the Q Values for each state with the index being (state, action)
        self.stateVisitedFreq = util.Counter() # Store the number of times a state is visited
        self.epsilon_decay_val = 0.99 # How much should the epsilon value decay by after each run
        
        # maintain scores
        self.score = 0
        self.lastMove = {
            "state":None,
            "action":None
        } # Storing the Last State and Action as a tuple (State, Action)


    # Accessor functions for the variable episodesSoFar controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value: float):
        self.epsilon = value

    def getAlpha(self) -> float:
        return self.alpha

    def setAlpha(self, value: float):
        self.alpha = value

    def getGamma(self) -> float:
        return self.gamma

    def getMaxAttempts(self) -> int:
        return self.maxAttempts

    def epsilon_decay(self):
        self.epsilon = self.epsilon * self.epsilon_decay_val


    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    @staticmethod
    def computeReward(startState: GameState,
                      endState: GameState) -> float:
        """
        Args:
            startState: A starting state
            endState: A resulting state

        Returns:
            The reward assigned for the given trajectory
        """
        
        return endState.getScore() - startState.getScore()


    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getQValue(self, state: GameStateFeatures, action: Directions) -> float:
        """
        Args:
            state: A given state
            action: Proposed action to take

        Returns:
            Q(state, action)
        """
        
        qValue = self.qTableValues[(str(state), action)]
        return qValue

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Args:
            state: The given state

        Returns:
            q_value: the maximum estimated Q-value attainable from the state
        """
        
        # Temporary array holding q values
        valueHolder = []
        legalActions = state.legal
        
        for m in legalActions:
            q = self.getQValue(state,m)
            valueHolder.append(q)
        
        if len(valueHolder) > 0:
            return max(valueHolder)
        else:
            return 0
        

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def learn(self,
              state: GameStateFeatures,
              action: Directions,
              reward: float,
              nextState: GameStateFeatures):
        """
        Performs a Q-learning update

        Args:
            state: the initial state
            action: the action that was took
            nextState: the resulting state
            reward: the reward received on this trajectory
        """
        
        # Update the Count for the number of visits of this State Action pair
        self.updateCount(state, action)

        currentQVal = self.getQValue(state, action)
        alphaVal = self.alpha
        maxQValNextState = self.maxQValue(nextState)
        innerVal = (reward + (self.gamma * maxQValNextState) - currentQVal)
        newQVal = currentQVal + (alphaVal * innerVal)

        # Update this states val
        self.qTableValues[str(state), action] = newQVal

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def updateCount(self,
                    state: GameStateFeatures,
                    action: Directions):
        """
        Updates the stored visitation counts.

        Args:
            state: Starting state
            action: Action taken
        """
        
        self.stateVisitedFreq[str(state), action] += 1 

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getCount(self,
                 state: GameStateFeatures,
                 action: Directions) -> int:
        """
        Args:
            state: Starting state
            action: Action taken

        Returns:
            Number of times that the action has been taken in a given state
        """
        
        return self.stateVisitedFreq[str(state), action]

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def explorationFn(self,
                      utility: float,
                      counts: int) -> float:
        """
        Computes exploration function.
        Return a value based on the counts

        HINT: Do a greed-pick or a least-pick

        Args:
            utility: expected utility for taking some action a in some given state s
            counts: counts for having taken visited

        Returns:
            The exploration value
        """
        
        
        #TODO: Fix this Func
        OPTIMISTIC_REWARD = sys.float_info.max
        if counts < self.maxAttempts:
            return OPTIMISTIC_REWARD / (counts + 1)
        else: 
            return utility
   


    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getAction(self, state: GameState) -> Directions:
        """
        Choose an action to take to maximise reward while
        balancing gathering data for learning

        If you wish to use epsilon-greedy exploration, implement it in this method.
        HINT: look at pacman_utils.util.flipCoin

        Args:
            state: the current state

        Returns:
            The action to take
        """
        # The data we have about the state of the game
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        print("Score: ", state.getScore())

        # Load the Game features for this state
        nextStateFeatures = GameStateFeatures(state)

        # If the previous state is not None. If it is not the starting state, then compute values for this run.
        if self.lastMove["state"] != None:
            previousStateFeatures = GameStateFeatures(self.lastMove["state"])
            reward = self.computeReward(self.lastMove["state"], state)
            self.learn(previousStateFeatures, self.lastMove["action"], reward, nextStateFeatures)            
        
        if random.random() < self.epsilon:
            nextAction = random.choice(legal)
        else:
            # Get the action with the maximum exploration value
            nextExploreActionPair = []
            for direction in legal:
                qValue = self.getQValue(nextStateFeatures, direction)
                stateActionCount = self.getCount(nextStateFeatures, direction)
                explorationVal = self.explorationFn(qValue, stateActionCount) 
                
                temp = (explorationVal, direction)
                nextExploreActionPair.append(temp)
            
            # Returns the tuple where the item at index 0 is max
            bestNextExploreActionPair = max(nextExploreActionPair)
            nextAction = bestNextExploreActionPair[1]
        
        self.epsilon_decay()
        # Epsilon value slowly decreases overtime
        self.epsilon = self.epsilon * 0.99


        # Rest the last State and Action values.
        self.lastMove["state"] = state
        self.lastMove["action"] = nextAction
        
        return nextAction
        

    def final(self, state: GameState):
        """
        Handle the end of episodes.
        This is called by the game after a win or a loss.

        Args:
            state: the final game state
        """
        # Register the Last iteration and Rest Last State and Action values to None for the next game
        nextState = GameStateFeatures(state)
        self.learn(GameStateFeatures(self.lastMove["state"]), self.lastMove["action"], self.computeReward(self.lastMove["state"], state), nextState)
        self.lastMove["state"] = None
        self.lastMove["action"] = None


        print(f"Game {self.getEpisodesSoFar()} just ended!")

        # Epsilon value slowly decreases overtime
        self.epsilon_decay()

        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg, '-' * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0)
