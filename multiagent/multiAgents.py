# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import sys
import warnings
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    # Check whether coordinates are inside the grid
    def isValid(self, x, y, grid):
        return x > 0 and x< grid.width - 1 and y > 0 and y < grid.height - 1

    # Find real distance and the path to the closest food to
    # pacman using bfs
    def closestFoodPath(self, newFood, newPos, walls):
        # Fringe
        q = util.Queue()
        path = []
        # Save initial state
        # State  is [x, y, list denoting path traversed so far]
        q.push([newPos[0], newPos[1], path])
        # Use set to check if position has been visited already
        been = set()
        while(not q.isEmpty()):
            cur = q.pop()
            x, y, path = cur[0], cur[1], cur[2]
            # Check if position has been visited already
            if (x, y) in been: continue
            been.add((x, y))
            # If there is a food on this position, we have found the answer
            if newFood[x][y]:
                return len(path), path
            # Check every possible move from the current position
            # (left, right, down, up)
            # For each one of them, check if we've been there, if the position
            # is within the map and if there is a wall there
            if (x - 1, y) not in been and self.isValid(x - 1, y, newFood) and not walls[x-1][y]:
                path.append((x-1, y))
                # Add new element to the fringe
                q.push([x-1, y, list(path)])
                path.pop()
            if (x + 1, y) not in been and self.isValid(x + 1, y, newFood) and not walls[x+1][y]:
                path.append((x+1, y))
                q.push([x+1, y, list(path)])
                path.pop()
            if (x, y - 1) not in been and self.isValid(x, y - 1, newFood) and not walls[x][y - 1]:
                path.append((x, y - 1))
                q.push([x, y - 1, list(path)])
                path.pop()
            if (x, y + 1) not in been and self.isValid(x, y + 1, newFood) and not walls[x][y + 1]:
                path.append((x, y + 1))
                q.push([x, y + 1, list(path)])
                path.pop()
        return -1, []

    # Check if any of the ghosts are on the passed position
    def checkForGhost(self, ghostStates, newPos):
        for ghost in ghostStates:
            gX, gY = ghost.getPosition()
            if gX == newPos[0] and gY == newPos[1]:
                return True 
        return False

    # Check if any of the ghosts are 1 move away from the passed position
    def checkForGhostClose(self, ghostStates, newPos):
        for ghost in ghostStates:
            gX, gY = ghost.getPosition()
            if (abs(gX - newPos[0]) + abs(gY - newPos[1])) == 1:
                return True 
        return False
            

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        walls = currentGameState.getWalls()
        # Time left before ghosts become dangerous again
        curScaredTimes = [ghostState.scaredTimer for ghostState in currentGameState.getGhostStates()]

        # Food map
        currentFood = currentGameState.getFood()
        score = 1000
        # Get real distance to the closest food
        nearestFoodDist, path = self.closestFoodPath(currentFood, newPos, walls)
        # Score is worse if distance to closest food is bigger
        score -= nearestFoodDist
        # If there is a food on the position, score is bigger
        if len(path) == 0 or path[0] == newPos:
            score += 5
        # If ghosts are scared...
        if min(curScaredTimes) == 0:
            # If there is a ghost on the new position, score is minimum
            if self.checkForGhost(newGhostStates, newPos):
                score = -sys.maxint - 1
            # If there is a ghost on the current position, score is minimum
            # Because pacman moves before ghosts
            if self.checkForGhost(currentGameState.getGhostStates(), newPos):
                score = -sys.maxint - 1
            # If there is a ghost really close to the new position, score is less
            if self.checkForGhostClose(currentGameState.getGhostStates(), newPos):
                score -= 10
        return score

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    # Get the value of the best action of min agent
    def getMin(self, gameState, ghostNum):
        # If game is finished, evaluate
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        minValue = sys.maxint
        
        # Get legal actions for the current ghost
        ghostActions = gameState.getLegalActions(ghostNum)
        # We need to check every possible state.
        # Generate them for the current ghost and then for each of them
        # call the next ghost.
        for gAction in ghostActions:
            nextState = gameState.generateSuccessor(ghostNum, gAction)
            # If the current ghost is the last...
            if ghostNum == gameState.getNumAgents() - 1:
                # If we aren't allowed to recurse deeper, just evaluate
                if self.depth == 0:
                    curVal = self.evaluationFunction(nextState)
                else:
                    # If we can go deeper, call the max agent
                    curVal, curAction = self.getMax(nextState)
            else:
                # Call the next ghost
                curVal = self.getMin(nextState, ghostNum + 1)
            # Memorize the best (minimum) value
            if curVal < minValue: minValue = curVal
        # Return the minimum value
        return minValue


    # Get the best action of max agent
    def getMax(self, gameState):
        # If game is finished, evaluate
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), Directions.STOP
        self.depth -= 1

        # Get legal actions for pacman
        pacActions = gameState.getLegalActions(0)
        maxAction = Directions.STOP
        maxValue = -sys.maxint - 1
        # Find the action which has the most value
        for pAction in pacActions:
            pacSuccessor = gameState.generateSuccessor(0, pAction)
            # Call min agent to recursively find the value of this action
            # Here, we call the function for the first ghost
            curVal = self.getMin(pacSuccessor, 1)
            # If the value for the current action is bigger than
            # already memorized one, memorize it and the action too
            if curVal > maxValue:
                maxValue = curVal
                maxAction = pAction
        
        self.depth += 1
        # Return value of the best action and the best action
        return maxValue, maxAction



    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        maxValue, maxAction = self.getMax(gameState)
        return maxAction
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    # Get the value of the best action of min agent
    # Works basically the same as minimax agent, but with alpha-beta pruning
    def getMin(self, gameState, ghostNum, a, b):
        minValue = sys.maxint
        # If game is finished, evaluate
        if gameState.isWin() or gameState.isLose():
            minValue = self.evaluationFunction(gameState)

        # Get legal actions for the current ghost
        ghostActions = gameState.getLegalActions(ghostNum)

        # We need to check every possible state.
        # Generate them for the current ghost and then for each of them
        # call the next ghost.
        for gAction in ghostActions:
            nextState = gameState.generateSuccessor(ghostNum, gAction)
            # If the current ghost is the last...
            if ghostNum == gameState.getNumAgents() - 1:
                # If we aren't allowed to recurse deeper, just evaluate
                if self.depth == 0:
                    curVal = self.evaluationFunction(nextState)
                else:
                    # If we can go deeper, call the max agent
                    curVal, curAction = self.getMax(nextState, a, b)
            else:
                # Call the next ghost
                curVal = self.getMin(nextState, ghostNum + 1, a, b)
            # If currently found minimum is less then the max value
            # Found by parent max agent call, we don't need to continue search
            if curVal < a: return curVal
            b = min(b, curVal)
            # Memorize the best (minimum) value
            if curVal < minValue: minValue = curVal
        return minValue



    # Get the best action of max agent
    # Works basically the same as minimax agent, but with alpha-beta pruning
    def getMax(self, gameState, a, b):
        maxAction = Directions.STOP
        maxValue = -sys.maxint - 1
        # If game is finished, evaluate
        if gameState.isWin() or gameState.isLose():
            maxValue = self.evaluationFunction(gameState)
        self.depth -= 1

        # Get legal actions for pacman
        pacActions = gameState.getLegalActions(0)
        # Find the action which has the most value
        for pAction in pacActions:
            pacSuccessor = gameState.generateSuccessor(0, pAction)
            # Call min agent to recursively find the value of this action
            # Here, we call the function for the first ghost
            curVal = self.getMin(pacSuccessor, 1, a, b)
            # If currently found maximum is more then the min value
            # Found by parent min agent call, we don't need to continue search
            if curVal > b:
                self.depth += 1
                return curVal, Directions.STOP
            a = max(a, curVal)
            # If the value for the current action is bigger than
            # already memorized one, memorize it and the action too
            if curVal > maxValue:
                maxValue = curVal
                maxAction = pAction
        
        self.depth += 1
        return maxValue, maxAction


    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        maxValue, maxAction = self.getMax(gameState, -sys.maxint - 1, sys.maxint)
        return maxAction
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    # Get the average value of the actions that the agent can take
    def getExpect(self, gameState, ghostNum):
        # If game is finished, evaluate
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        sumVal = 0
        
        # Get legal actions for ghost
        ghostActions = gameState.getLegalActions(ghostNum)
        for gAction in ghostActions:
            nextState = gameState.generateSuccessor(ghostNum, gAction)
            # If the current ghost is the last...
            if ghostNum == gameState.getNumAgents() - 1:
                # If we aren't allowed to recurse deeper, just evaluate
                if self.depth == 0:
                    curVal = self.evaluationFunction(nextState)
                else:
                    # If we can go deeper, call the max agent
                    curVal, curAction = self.getMax(nextState)
            else:
                # Call the next ghost
                curVal = self.getExpect(nextState, ghostNum + 1)
            sumVal += curVal
        # Return the average value of the actions that the agent can take
        return (sumVal * 1.0)/len(ghostActions)



    # Get the best action of max agent
    def getMax(self, gameState):
        # If game is finished, evaluate
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), Directions.STOP
        self.depth -= 1

        # Get legal actions for pacman
        pacActions = gameState.getLegalActions(0)
        maxAction = Directions.STOP
        maxValue = -sys.maxint - 1
        # Find the action which has the most value
        for pAction in pacActions:
            pacSuccessor = gameState.generateSuccessor(0, pAction)
            # Call expect agent to recursively find the value of this action
            # Here, we call the function for the first ghost
            curVal = self.getExpect(pacSuccessor, 1)
            # If the value for the current action is bigger than
            # already memorized one, memorize it and the action too
            if curVal > maxValue:
                maxValue = curVal
                maxAction = pAction
        
        self.depth += 1
        return maxValue, maxAction

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        maxValue, maxAction = self.getMax(gameState)
        return maxAction
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """

    # Current pacman position
    pacPos = currentGameState.getPacmanPosition()
    # Current map of food
    mapFood = currentGameState.getFood()

    walls = currentGameState.getWalls()

    # Time left before ghosts become dangerous again
    curScaredTimes = [ghostState.scaredTimer for ghostState in currentGameState.getGhostStates()]

    # Find the real distance to the closest food
    closestFoodDist, path = closestFoodPath(mapFood, pacPos, walls)
    # Find the manhattan distance to the closest ghost
    closestGhostDist = closestGhostManh(currentGameState.getGhostPositions(), pacPos)
    eatGhost = 0
    # If ghosts are scared...
    if min(curScaredTimes) > 0:
        # Bonus points if scared ghost is closer. Eating ghost = more points
        eatGhost = 30 - closestGhostDist
        # Score shouldn't be better for being further from the closest ghost
        closestGhostDist = 0
    return currentGameState.getScore() - closestFoodDist + closestGhostDist + eatGhost
    util.raiseNotDefined()

# Returns manhattan distance between pacman and closest ghost
def closestGhostManh(ghosts, pacPos):
    minVal = sys.maxint
    for ghost in ghosts:
        curVal = abs(ghost[0] - pacPos[0]) + abs(ghost[1] - pacPos[1])
        minVal = min(minVal, curVal)
    return minVal

# Check whether coordinates are inside the grid
def isValid(x, y, grid):
    return x > 0 and x< grid.width - 1 and y > 0 and y < grid.height - 1

# Find real distance and the path to the closest food to
# pacman using bfs
def closestFoodPath(newFood, newPos, walls):
    # Fringe
    q = util.Queue()
    path = []
    # Save initial state
    # State  is [x, y, list denoting path traversed so far]
    q.push([newPos[0], newPos[1], path])
    # Use set to check if position has been visited already
    been = set()
    while(not q.isEmpty()):
        cur = q.pop()
        x, y, path = cur[0], cur[1], cur[2]
        # Check if position has been visited already
        if (x, y) in been: continue
        been.add((x, y))
        # If there is a food on this position, we have found the answer
        if newFood[x][y]:
            return len(path), path
        # Check every possible move from the current position
        # (left, right, down, up)
        # For each one of them, check if we've been there, if the position
        # is within the map and if there is a wall there
        if (x - 1, y) not in been and isValid(x - 1, y, newFood) and not walls[x-1][y]:
            path.append((x-1, y))
            # Add new element to the fringe
            q.push([x-1, y, list(path)])
            path.pop()
        if (x + 1, y) not in been and isValid(x + 1, y, newFood) and not walls[x+1][y]:
            path.append((x+1, y))
            q.push([x+1, y, list(path)])
            path.pop()
        if (x, y - 1) not in been and isValid(x, y - 1, newFood) and not walls[x][y - 1]:
            path.append((x, y - 1))
            q.push([x, y - 1, list(path)])
            path.pop()
        if (x, y + 1) not in been and isValid(x, y + 1, newFood) and not walls[x][y + 1]:
            path.append((x, y + 1))
            q.push([x, y + 1, list(path)])
            path.pop()
    return -1, []

# Abbreviation
better = betterEvaluationFunction

