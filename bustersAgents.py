# bustersAgents.py
# ----------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import util
from game import Agent
from game import Directions
from keyboardAgents import KeyboardAgent
from wekaI import Weka
import inference
import busters
import os

last_move = "Stop"
prevState = []
predictN = 1
distWest = 0
distEast = 0
distNorth = 0
distSouth = 0

class NullGraphics:
    "Placeholder for graphics"
    def initialize(self, state, isBlue = False):
        pass
    def update(self, state):
        pass
    def pause(self):
        pass
    def draw(self, state):
        pass
    def updateDistributions(self, dist):
        pass
    def finish(self):
        pass

class KeyboardInference(inference.InferenceModule):
    """
    Basic inference module for use with the keyboard.
    """
    def initializeUniformly(self, gameState):
        "Begin with a uniform distribution over ghost positions."
        self.beliefs = util.Counter()
        for p in self.legalPositions: self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observe(self, observation, gameState):
        noisyDistance = observation
        emissionModel = busters.getObservationDistribution(noisyDistance)
        pacmanPosition = gameState.getPacmanPosition()
        allPossible = util.Counter()
        for p in self.legalPositions:
            trueDistance = util.manhattanDistance(p, pacmanPosition)
            if emissionModel[trueDistance] > 0:
                allPossible[p] = 1.0
        allPossible.normalize()
        self.beliefs = allPossible

    def elapseTime(self, gameState):
        pass

    def getBeliefDistribution(self):
        return self.beliefs


class BustersAgent:
    "An agent that tracks and displays its beliefs about ghost positions."

    def __init__( self, index = 0, inference = "ExactInference", ghostAgents = None, observeEnable = True, elapseTimeEnable = True):
        inferenceType = util.lookup(inference, globals())
        self.inferenceModules = [inferenceType(a) for a in ghostAgents]
        self.observeEnable = observeEnable
        self.elapseTimeEnable = elapseTimeEnable
        global predictN
        if predictN <= 0:
            predictN = 1
        # Iniciamos una maquina virtual de Java para Weka
        self.weka = Weka()
        self.weka.start_jvm()
        
    def registerInitialState(self, gameState):
        "Initializes beliefs and inference modules"
        import __main__
        self.display = __main__._display
        for inference in self.inferenceModules:
            inference.initialize(gameState)
        self.ghostBeliefs = [inf.getBeliefDistribution() for inf in self.inferenceModules]
        self.firstMove = True

    def observationFunction(self, gameState):
        "Removes the ghost states from the gameState"
        agents = gameState.data.agentStates
        gameState.data.agentStates = [agents[0]] + [None for i in range(1, len(agents))]
        return gameState

    def getAction(self, gameState):
        "Updates beliefs, then chooses an action based on updated beliefs."
        #for index, inf in enumerate(self.inferenceModules):
        #    if not self.firstMove and self.elapseTimeEnable:
        #        inf.elapseTime(gameState)
        #    self.firstMove = False
        #    if self.observeEnable:
        #        inf.observeState(gameState)
        #    self.ghostBeliefs[index] = inf.getBeliefDistribution()
        #self.display.updateDistributions(self.ghostBeliefs)
        return self.chooseAction(gameState)

    def chooseAction(self, gameState):
        "By default, a BustersAgent just stops.  This should be overridden."
        return Directions.STOP

class BustersKeyboardAgent(BustersAgent, KeyboardAgent):
    "An agent controlled by the keyboard that displays beliefs about ghost positions."

    def __init__(self, index = 0, inference = "KeyboardInference", ghostAgents = None):
        KeyboardAgent.__init__(self, index)
        BustersAgent.__init__(self, index, inference, ghostAgents)
        self.countActions = 0

    def getAction(self, gameState):
        return BustersAgent.getAction(self, gameState)

    def chooseAction(self, gameState):
        global last_move
        global distWest
        global distEast
        global distNorth
        global distSouth

        distWest = 99999
        distEast = 99999
        distNorth = 99999
        distSouth = 99999

        self.distancer = Distancer(gameState.data.layout, False)
        self.countActions = self.countActions + 1
        last_move = KeyboardAgent.getAction(self, gameState)

        # Almacenamos en variables una serie de datos utiles
        legal = gameState.getLegalActions(0) ##Legal position from the pacman
        posPacman = gameState.getPacmanPosition()
        walls = gameState.getWalls()
        livingGhosts = gameState.getLivingGhosts()
        #move NORTH
        if Directions.NORTH in legal:
            # Inicia un contador a 1 para no tener el cuenta el pacman
            iterator = 1
            # Almacena la posicion del pacman resultante de ejecutar la accion
            buffPacman = posPacman[0], posPacman[1] + 1
            # Comprueba que la casilla objetivo no contenga un muro
            if walls[buffPacman[0]][buffPacman[1]] == False:
                # Itera sobre los fantasmas
                for g in gameState.getGhostPositions():
                    # Comprueba que los fantasmas estan vivos
                    if livingGhosts[iterator] == True:
                        # Si la distancia minima actual es menor que la almacenada, se sobreescribe
                        if self.distancer.getDistance(g, buffPacman) < distNorth:
                            distNorth = self.distancer.getDistance(g, buffPacman)
                    iterator += 1

        #move SOUTH
        if Directions.SOUTH in legal:
            # Inicia un contador a 1 para no tener el cuenta el pacman
            iterator = 1
            # Almacena la posicion del pacman resultante de ejecutar la accion
            buffPacman = posPacman[0], posPacman[1] - 1
            # Comprueba que la casilla objetivo no contenga un muro
            if walls[buffPacman[0]][buffPacman[1]] == False:
                # Itera sobre los fantasmas
                for g in gameState.getGhostPositions():
                    # Comprueba que los fantasmas estan vivos
                    if livingGhosts[iterator] == True:
                        # Si la distancia minima actual es menor que la almacenada, se sobreescribe
                        if self.distancer.getDistance(g, buffPacman) < distSouth:
                            distSouth = self.distancer.getDistance(g, buffPacman)
                    iterator += 1
                        
        #move EAST
        if Directions.EAST in legal:
            # Inicia un contador a 1 para no tener el cuenta el pacman
            iterator = 1
            # Almacena la posicion del pacman resultante de ejecutar la accion
            buffPacman = posPacman[0] + 1, posPacman[1]
            # Comprueba que la casilla objetivo no contenga un muro
            if walls[buffPacman[0]][buffPacman[1]] == False:
                # Itera sobre los fantasmas
                for g in gameState.getGhostPositions():
                    # Comprueba que los fantasmas estan vivos
                    if livingGhosts[iterator] == True:
                        # Si la distancia minima actual es menor que la almacenada, se sobreescribe
                        if self.distancer.getDistance(g, buffPacman) < distEast:
                            distEast = self.distancer.getDistance(g, buffPacman)
                    iterator += 1
                       
        #move WEST
        if Directions.WEST in legal:
            # Inicia un contador a 1 para no tener el cuenta el pacman
            iterator = 1
            # Almacena la posicion del pacman resultante de ejecutar la accion
            buffPacman = posPacman[0] - 1, posPacman[1]
            # Comprueba que la casilla objetivo no contenga un muro
            if walls[buffPacman[0]][buffPacman[1]] == False:
                # Itera sobre los fantasmas
                for g in gameState.getGhostPositions():
                    # Comprueba que los fantasmas estan vivos
                    if livingGhosts[iterator] == True:
                        # Si la distancia minima actual es menor que la almacenada, se sobreescribe
                        if self.distancer.getDistance(g, buffPacman) < distWest:
                            distWest = self.distancer.getDistance(g, buffPacman)
                    iterator += 1

        return last_move

    def printLineData(self, gameState):
        # En esta funcion usaremos 2 variables globales
        global predictN
        global prevState
        
        # Intentamos abrir el archivo de salida, si no existe, lo creamos usando createWekaFile()
        if(os.path.isfile("training_tutorial1.arff") == False):
            attributesList = [["distNorth","NUMERIC"],["distSouth","NUMERIC"],["distEast","NUMERIC"],["distWest","NUMERIC"], 
            ["Score","NUMERIC"],["NextScore","NUMERIC"],["NearestFood","NUMERIC"],["lastMove","{North,South,East,West,Stop}"]]
            self.createWekaFile(attributesList)

        # Necesitamos la funcion distancer para hacer calculos de distancias
        self.distancer = Distancer(gameState.data.layout, False)

        # Si hemos llegado al turno N, imprimimos en el archivo
        if self.countActions > predictN:
            # Cada vez vamos a sacar 9 elementos de la lista (8 atributos + "\n")
            counter = 9
            # Abrimos el archivo de salida
            file = open("training_tutorial1.arff", "a")
            # Obtenemos la puntuacion actual, para usarla de valor predecido
            prevState[5] = gameState.getScore()
            while(counter > 0):
                # Usamos pop para sacar uno a uno los elementos de la lista y escribirlos en el archivo de salida
                x = prevState.pop(0);
                file.write("%s" % (x))
                # Imprimimos comas entre atributos excepto entre los 2 ultimos (last_move y "\n")
                if counter > 2:
                    file.write(",")
                counter -= 1
            # Cerramos el archivo de salida
            file.close()
        # Metemos en la lista las variables de distancia
        prevState.append(distNorth)
        prevState.append(distSouth)
        prevState.append(distEast)
        prevState.append(distWest)
        # Metemos en la lista la puntuacion actual, y un placeholder para nextScore
        prevState.append(gameState.getScore())
        prevState.append(gameState.getScore()-1)
        # Metemos en la lista la distancia a la comida mas cercana, si no hay, metemos 99999 en su lugar
        if (gameState.getDistanceNearestFood() == None):
            prevState.append(99999)
        else:
            prevState.append(gameState.getDistanceNearestFood())
        # Introducimos el movimiento realizado, que se ha obtenido durante chooseAction()
        prevState.append(last_move)
        prevState.append("\n")


    def createWekaFile(self, attributesList):
        # Abrimos el archivo en modo append, para que no se sobreescriba el archivo
        file = open("training_tutorial1.arff", "a")
        # Escribimos la cabecera del archivo
        file.write("@RELATION 'training_tutorial1'\n\n")
        # Escribimos todos los atributos
        for l in attributesList:
            file.write("@ATTRIBUTE %s %s\n" % (l[0], l[1]))
        # Escribimos el indicador de que empiezan los datos
        file.write("\n@data\n")

from distanceCalculator import Distancer
from game import Actions
from game import Directions
import random, sys

'''Random PacMan Agent'''
class RandomPAgent(BustersAgent):
    
    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        
    ''' Example of counting something'''
    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if(height == True):
                    food = food + 1
        return food
    
    ''' Print the layout'''  
    def printGrid(self, gameState):
        table = ""
        ##print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table
        
    def chooseAction(self, gameState):
        move = Directions.STOP
        legal = gameState.getLegalActions(0) ##Legal position from the pacman
        move_random = random.randint(0, 3)
        if   ( move_random == 0 ) and Directions.WEST in legal:  move = Directions.WEST
        if   ( move_random == 1 ) and Directions.EAST in legal: move = Directions.EAST
        if   ( move_random == 2 ) and Directions.NORTH in legal:   move = Directions.NORTH
        if   ( move_random == 3 ) and Directions.SOUTH in legal: move = Directions.SOUTH
        return move
        
        
class GreedyBustersAgent(BustersAgent):
    "An agent that charges the closest ghost."

    def registerInitialState(self, gameState):
        "Pre-computes the distance between every two points."
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)

    def chooseAction(self, gameState):
        """
        First computes the most likely position of each ghost that has
        not yet been captured, then chooses an action that brings
        Pacman closer to the closest ghost (according to mazeDistance!).

        To find the mazeDistance between any two positions, use:
          self.distancer.getDistance(pos1, pos2)

        To find the successor position of a position after an action:
          successorPosition = Actions.getSuccessor(position, action)

        livingGhostPositionDistributions, defined below, is a list of
        util.Counter objects equal to the position belief
        distributions for each of the ghosts that are still alive.  It
        is defined based on (these are implementation details about
        which you need not be concerned):

          1) gameState.getLivingGhosts(), a list of booleans, one for each
             agent, indicating whether or not the agent is alive.  Note
             that pacman is always agent 0, so the ghosts are agents 1,
             onwards (just as before).

          2) self.ghostBeliefs, the list of belief distributions for each
             of the ghosts (including ghosts that are not alive).  The
             indices into this list should be 1 less than indices into the
             gameState.getLivingGhosts() list.
        """
        pacmanPosition = gameState.getPacmanPosition()
        legal = [a for a in gameState.getLegalPacmanActions()]
        livingGhosts = gameState.getLivingGhosts()
        livingGhostPositionDistributions = \
            [beliefs for i, beliefs in enumerate(self.ghostBeliefs)
             if livingGhosts[i+1]]
        return Directions.EAST



class BasicAgentAA(BustersAgent):
    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        self.countActions = 0
        
    ''' Example of counting something'''
    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if(height == True):
                    food = food + 1
        return food
    
    ''' Print the layout'''  
    def printGrid(self, gameState):
        table = ""
        #print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table

    def printInfo(self, gameState):
        print "---------------- TICK ", self.countActions, " --------------------------"
        # Dimensiones del mapa
        width, height = gameState.data.layout.width, gameState.data.layout.height
        print "Width: ", width, " Height: ", height
        # Posicion del Pacman
        print "Pacman position: ", gameState.getPacmanPosition()
        # Acciones legales de pacman en la posicion actual
        print "Legal actions: ", gameState.getLegalPacmanActions()
        # Direccion de pacman
        print "Pacman direction: ", gameState.data.agentStates[0].getDirection()
        # Numero de fantasmas
        print "Number of ghosts: ", gameState.getNumAgents() - 1
        # Fantasmas que estan vivos (el indice 0 del array que se devuelve corresponde a pacman y siempre es false)
        print "Living ghosts: ", gameState.getLivingGhosts()
        # Posicion de los fantasmas
        print "Ghosts positions: ", gameState.getGhostPositions()
        # Direciones de los fantasmas
        print "Ghosts directions: ", [gameState.getGhostDirections().get(i) for i in range(0, gameState.getNumAgents() - 1)]
        # Distancia de manhattan a los fantasmas
        print "Ghosts distances: ", gameState.data.ghostDistances
        # Puntos de comida restantes
        print "Pac dots: ", gameState.getNumFood()
        # Distancia de manhattan a la comida mas cercada
        print "Distance nearest pac dots: ", gameState.getDistanceNearestFood()
        # Paredes del mapa
        print "Map:  \n", gameState.getWalls()
        # Puntuacion
        print "Score: ", gameState.getScore()
        
    def chooseAction(self, gameState):

        # En esta funcion usaremos 5 variables globales, una para la accion, y 4 para las distancias
        global last_move
        global distWest
        global distEast
        global distNorth
        global distSouth
        '''
        # Incrementamos el contador de turnos
        self.countActions = self.countActions + 1
        # Se imprimen datos relevantes sobre la partida
        self.printInfo(gameState)
        # Por defecto, el movimiento a ejecutar es "Stop"
        move = Directions.STOP
        # Se inicializan las distancias a un numero muy alto
        distWest = 99999
        distEast = 99999
        distNorth = 99999
        distSouth = 99999

        # Almacenamos en variables una serie de datos relevantes
        legal = gameState.getLegalActions(0) ##Legal position from the pacman
        posPacman = gameState.getPacmanPosition()
        minDist = 99999
        walls = gameState.getWalls()
        livingGhosts = gameState.getLivingGhosts()
        #move NORTH
        if Directions.NORTH in legal:
            # Inicia un contador a 1 para no tener el cuenta el pacman
            iterator = 1
            # Almacena la posicion del pacman resultante de ejecutar la accion
            buffPacman = posPacman[0], posPacman[1] + 1
            # Comprueba que la casilla objetivo no contenga un muro
            if walls[buffPacman[0]][buffPacman[1]] == False:
                # Itera sobre los fantasmas
                for x in gameState.getGhostPositions():
                    # Comprueba que los fantasmas estan vivos
                    if livingGhosts[iterator] == True:
                        # Si la distancia actual es menor o igual que la previa de este movimiento, se sobreescribe
                        if self.distancer.getDistance(x, buffPacman) <= distNorth:
                            distNorth = self.distancer.getDistance(x, buffPacman)
                            # Se comprueba si la distancia es menor que la minima de todas las acciones
                            if distNorth < minDist:
                                # Se sobreescribe y se cambia el movimiento a realizar
                                minDist = distNorth
                                move = Directions.NORTH
                            # Si la distancia es igual a la minima actual, se comprueba si la casilla nueva contiene comida
                            # de ser asi, se cambia el movimiento a realizar
                            # Esto sirve para que en caso de haber 2 acciones igual de buenas, elija la que tiene comida
                            elif distNorth == minDist:
                                if gameState.hasFood(buffPacman[0],buffPacman[1]):
                                    move = Directions.NORTH
                    iterator = iterator + 1
        #move SOUTH
        if Directions.SOUTH in legal:
            # Inicia un contador a 1 para no tener el cuenta el pacman
            iterator = 1
            # Almacena la posicion del pacman resultante de ejecutar la accion
            buffPacman = posPacman[0], posPacman[1] - 1
            # Comprueba que la casilla objetivo no contenga un muro
            if walls[buffPacman[0]][buffPacman[1]] == False:
                # Itera sobre los fantasmas
                for x in gameState.getGhostPositions():
                    # Comprueba que los fantasmas estan vivos
                    if livingGhosts[iterator] == True:
                        # Si la distancia actual es menor o igual que la previa de este movimiento, se sobreescribe
                         if self.distancer.getDistance(x, buffPacman) <= distSouth:
                            distSouth = self.distancer.getDistance(x, buffPacman)
                            # Se comprueba si la distancia es menor que la minima de todas las acciones
                            if distSouth < minDist:
                                # Se sobreescribe y se cambia el movimiento a realizar
                                minDist = distSouth
                                move = Directions.SOUTH
                            # Si la distancia es igual a la minima actual, se comprueba si la casilla nueva contiene comida
                            # de ser asi, se cambia el movimiento a realizar
                            # Esto sirve para que en caso de haber 2 acciones igual de buenas, elija la que tiene comida
                            elif distSouth == minDist:
                                if gameState.hasFood(buffPacman[0],buffPacman[1]):
                                    move = Directions.SOUTH
                    iterator = iterator + 1
        #move EAST
        if Directions.EAST in legal:
            # Inicia un contador a 1 para no tener el cuenta el pacman
            iterator = 1
            # Almacena la posicion del pacman resultante de ejecutar la accion
            buffPacman = posPacman[0] + 1, posPacman[1]
            # Comprueba que la casilla objetivo no contenga un muro
            if walls[buffPacman[0]][buffPacman[1]] == False:
                # Itera sobre los fantasmas
                for x in gameState.getGhostPositions():
                    # Comprueba que los fantasmas estan vivos
                    if livingGhosts[iterator] == True:
                        # Si la distancia actual es menor o igual que la previa de este movimiento, se sobreescribe
                        if self.distancer.getDistance(x, buffPacman) <= distEast:
                            distEast = self.distancer.getDistance(x, buffPacman)
                            # Se comprueba si la distancia es menor que la minima de todas las acciones
                            if distEast < minDist:
                                # Se sobreescribe y se cambia el movimiento a realizar
                                minDist = distEast
                                move = Directions.EAST
                            # Si la distancia es igual a la minima actual, se comprueba si la casilla nueva contiene comida
                            # de ser asi, se cambia el movimiento a realizar
                            # Esto sirve para que en caso de haber 2 acciones igual de buenas, elija la que tiene comida
                            elif distEast == minDist:
                                if gameState.hasFood(buffPacman[0],buffPacman[1]):
                                    move = Directions.EAST
                    iterator = iterator + 1
        #move WEST
        if Directions.WEST in legal:
            # Inicia un contador a 1 para no tener el cuenta el pacman
            iterator = 1
            # Almacena la posicion del pacman resultante de ejecutar la accion
            buffPacman = posPacman[0] - 1, posPacman[1]
            # Comprueba que la casilla objetivo no contenga un muro
            if walls[buffPacman[0]][buffPacman[1]] == False:
                # Itera sobre los fantasmas
                for x in gameState.getGhostPositions():
                    # Comprueba que los fantasmas estan vivos
                    if livingGhosts[iterator] == True:
                        # Si la distancia actual es menor o igual que la previa de este movimiento, se sobreescribe
                        if self.distancer.getDistance(x, buffPacman) <= distWest:
                            distWest = self.distancer.getDistance(x, buffPacman)
                            # Se comprueba si la distancia es menor que la minima de todas las acciones
                            if distWest < minDist:
                                # Se sobreescribe y se cambia el movimiento a realizar
                                minDist = distWest
                                move = Directions.WEST
                            # Si la distancia es igual a la minima actual, se comprueba si la casilla nueva contiene comida
                            # de ser asi, se cambia el movimiento a realizar
                            # Esto sirve para que en caso de haber 2 acciones igual de buenas, elija la que tiene comida
                            elif distWest == minDist:
                                if gameState.hasFood(buffPacman[0],buffPacman[1]):
                                    move = Directions.WEST
                    iterator = iterator + 1
        # Se almacena last_move para usarla en otras funciones
        last_move = move
        return move
        '''
        # Inicializamos la lista que le pasaremos a Weka
        x = []
        # Inicializamos las variables de distancia a un numero muy alto, que significa que la accion es ilegal
        distWest = 99999
        distEast = 99999
        distNorth = 99999
        distSouth = 99999
        
        # Almacenamos en variables una serie de datos utiles
        legal = gameState.getLegalActions(0) ##Legal position from the pacman
        posPacman = gameState.getPacmanPosition()
        walls = gameState.getWalls()
        livingGhosts = gameState.getLivingGhosts()
        #move NORTH
        if Directions.NORTH in legal:
            # Inicia un contador a 1 para no tener el cuenta el pacman
            iterator = 1
            # Almacena la posicion del pacman resultante de ejecutar la accion
            buffPacman = posPacman[0], posPacman[1] + 1
            # Comprueba que la casilla objetivo no contenga un muro
            if walls[buffPacman[0]][buffPacman[1]] == False:
                # Itera sobre los fantasmas
                for g in gameState.getGhostPositions():
                    # Comprueba que los fantasmas estan vivos
                    if livingGhosts[iterator] == True:
                        # Si la distancia minima actual es menor que la almacenada, se sobreescribe
                        if self.distancer.getDistance(g, buffPacman) < distNorth:
                            distNorth = self.distancer.getDistance(g, buffPacman)
                    iterator += 1

        #move SOUTH
        if Directions.SOUTH in legal:
            # Inicia un contador a 1 para no tener el cuenta el pacman
            iterator = 1
            # Almacena la posicion del pacman resultante de ejecutar la accion
            buffPacman = posPacman[0], posPacman[1] - 1
            # Comprueba que la casilla objetivo no contenga un muro
            if walls[buffPacman[0]][buffPacman[1]] == False:
                # Itera sobre los fantasmas
                for g in gameState.getGhostPositions():
                    # Comprueba que los fantasmas estan vivos
                    if livingGhosts[iterator] == True:
                        # Si la distancia minima actual es menor que la almacenada, se sobreescribe
                        if self.distancer.getDistance(g, buffPacman) < distSouth:
                            distSouth = self.distancer.getDistance(g, buffPacman)
                    iterator += 1
                        
        #move EAST
        if Directions.EAST in legal:
            # Inicia un contador a 1 para no tener el cuenta el pacman
            iterator = 1
            # Almacena la posicion del pacman resultante de ejecutar la accion
            buffPacman = posPacman[0] + 1, posPacman[1]
            # Comprueba que la casilla objetivo no contenga un muro
            if walls[buffPacman[0]][buffPacman[1]] == False:
                # Itera sobre los fantasmas
                for g in gameState.getGhostPositions():
                    # Comprueba que los fantasmas estan vivos
                    if livingGhosts[iterator] == True:
                        # Si la distancia minima actual es menor que la almacenada, se sobreescribe
                        if self.distancer.getDistance(g, buffPacman) < distEast:
                            distEast = self.distancer.getDistance(g, buffPacman)
                    iterator += 1
                       
        #move WEST
        if Directions.WEST in legal:
            # Inicia un contador a 1 para no tener el cuenta el pacman
            iterator = 1
            # Almacena la posicion del pacman resultante de ejecutar la accion
            buffPacman = posPacman[0] - 1, posPacman[1]
            # Comprueba que la casilla objetivo no contenga un muro
            if walls[buffPacman[0]][buffPacman[1]] == False:
                # Itera sobre los fantasmas
                for g in gameState.getGhostPositions():
                    # Comprueba que los fantasmas estan vivos
                    if livingGhosts[iterator] == True:
                        # Si la distancia minima actual es menor que la almacenada, se sobreescribe
                        if self.distancer.getDistance(g, buffPacman) < distWest:
                            distWest = self.distancer.getDistance(g, buffPacman)
                    iterator += 1

        # Metemos en la lista las variables de distancia
        x.append(distNorth)
        x.append(distSouth)
        x.append(distEast)
        x.append(distWest)
        # -------------------------- COMMENT THIS PART IF USING NoFood_NoScore --------------------------
        # Metemos en la lista la puntuacion
        x.append(gameState.getScore())
        # Metemos en la lista la distancia a la comida mas cercana, si no hay, metemos 99999 en su lugar
        if (gameState.getDistanceNearestFood() == None):
            x.append(99999)
        else:
            x.append(gameState.getDistanceNearestFood())
        # -----------------------------------------------------------------------------------------------

        # Pasamos los datos necesarios a Weka para que los clasifique
        a = self.weka.predict("./Models/Classification/Tutorial1/Samemaps/Unfiltered/LMT.model", x, "./Training/training_tutorial1_noNextScore.arff")
        
        # ------ Estas lineas sirven para evitar que el agente automatico entre en un bucle infinito ----------
        # El bug se produce cuando no se puede hacer la accion "East" y la accion "South" es mejor que "North"
        """ if (distEast == 99999 and distNorth > distSouth and a == 'North'):
            a = 'South' """
        # -----------------------------------------------------------------------------------------------------

        # ------ Estas lineas sirven para que el agente automatico de teclado no haga acciones ilegales -------
        """ if (a == 'East' and distEast == 99999):
            a = 'Stop'
        elif (a == 'West' and distWest == 99999):
            a = 'Stop'
        elif (a == 'North' and distNorth == 99999):
            a = 'Stop'
        elif (a == 'South' and distSouth == 99999):
            a = 'Stop' """
        # ------------------------------------------------------------------------------------------------------
        # Se almacena last_move para usarla en otras funciones
        last_move = a
        return a
        

    def printLineData(self, gameState):
        # En esta funcion usaremos 2 variables globales
        global predictN
        global prevState
        
        # Intentamos abrir el archivo de salida, si no existe, lo creamos usando createWekaFile()
        if(os.path.isfile("training_tutorial1.arff") == False):
            attributesList = [["distNorth","NUMERIC"],["distSouth","NUMERIC"],["distEast","NUMERIC"],["distWest","NUMERIC"], 
            ["Score","NUMERIC"],["NextScore","NUMERIC"],["NearestFood","NUMERIC"],["lastMove","{North,South,East,West,Stop}"]]
            self.createWekaFile(attributesList)

        # Si hemos llegado al turno N, imprimimos en el archivo
        if self.countActions > predictN:
            # Cada vez vamos a sacar 9 elementos de la lista (8 atributos + "\n")
            counter = 9
            # Abrimos el archivo de salida
            file = open("training_tutorial1.arff", "a")
            # Obtenemos la puntuacion actual, para usarla de valor predecido
            prevState[5] = gameState.getScore()
            while(counter > 0):
                # Usamos pop para sacar uno a uno los elementos de la lista y escribirlos en el archivo de salida
                x = prevState.pop(0);
                file.write("%s" % (x))
                # Imprimimos comas entre atributos excepto entre los 2 ultimos (last_move y "\n")
                if counter > 2:
                    file.write(",")
                counter -= 1
            # Cerramos el archivo de salida
            file.close()
        # Metemos en la lista las variables de distancia
        prevState.append(distNorth)
        prevState.append(distSouth)
        prevState.append(distEast)
        prevState.append(distWest)
        # Metemos en la lista la puntuacion actual, y un placeholder para nextScore
        prevState.append(gameState.getScore())
        prevState.append(gameState.getScore()-1)
        # Metemos en la lista la distancia a la comida mas cercana, si no hay, metemos 99999 en su lugar
        if (gameState.getDistanceNearestFood() == None):
            prevState.append(99999)
        else:
            prevState.append(gameState.getDistanceNearestFood())
        # Introducimos el movimiento realizado, que se ha obtenido durante chooseAction()
        prevState.append(last_move)
        prevState.append("\n")

    def createWekaFile(self, attributesList):
        # Abrimos el archivo en modo append, para que no se sobreescriba el archivo
        file = open("training_tutorial1.arff", "a")
        # Escribimos la cabecera del archivo
        file.write("@RELATION 'training_tutorial1'\n\n")
        # Escribimos todos los atributos
        for l in attributesList:
            file.write("@ATTRIBUTE %s %s\n" % (l[0], l[1]))
        # Escribimos el indicador de que empiezan los datos
        file.write("\n@data\n")