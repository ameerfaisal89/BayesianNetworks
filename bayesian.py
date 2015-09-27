# -*- coding: utf-8 -*-
"""
Created on Fri May 15 21:19:51 2015

@author: Ameer Asif Khan
"""
import graph;
import numpy as np;

class BayesianNet( graph.DirectedGraph ):
    '''
    A Bayesian Network. Inherits from the DirectedGraph class.
    evidenceList is the list of tuple pairs, specifying nodes on which
    evidence is set, and the state values.
    '''
    def __init__( self, *args, **kwargs ):
        '''
        Initializes the Bayesian network. evidenceList is set to empty list.
        '''
        super( ).__init__( *args, **kwargs );
        self.evidenceList = [ ];
    
    def addNode( self, nName ):
        '''
        Adds a node to the network. No action occurs if the node already exists.
        
        @param nName Name of the node to be added to the network.
        '''
        self.addVertex( nName );
    
    def addChild( self, pName, cName ):
        '''
        Adds a parent and child node and a directed edge from the parent to child. If any of the
        two nodes do not exist, they are automatically created.
        
        @param pName Name of the parent node
        @param cName Name of the child node
        '''
        self.addEdge( pName, cName );
    
    def getParents( self, uName ):
        '''
        Returns the parent nodes for a specified node.
        
        @param uName Name of the node whose parent nodes are to be determined.
        @return Set of _Vertex objects that are parent nodes of the given node uName.
        '''
        u = self.vertexObject( uName );
        parents = { v for v in self.vertexes( ) if u in v.neighbors };
        return parents;
    
    def addProbabilityTable( self, uName, probability, states, dependencyList = None ):
        '''
        Adds probability information to a node. The network must be created before calling
        this method on the each node. The probability specified must be either the marginal
        probability table for independent nodes, and conditional probability table for dependent
        nodes. The dependency list is also to be specified for dependent nodes, which is in the
        same order as the dimensions of the conditional probability table.
        
        @param uName Name of the node to add probability information
        @param probability The probability matrix as a numpy array. Marginal for independent nodes
               and conditional for dependent nodes
        @param states List of the state values the node takes
        @param dependencyList List of node names in the order corresponding to the dimensions of
               the probability matrix. dependencyList is None for independent nodes.
        '''
        if ( dependencyList is None ):
            dependencyList = [ ];
        
        shape = np.shape( probability );
        
        if ( shape[ 0 ] != len( states ) ):
            raise ValueError( "Incorrect states" );
        
        if ( len( shape ) != len( self.getParents( uName ) ) + 1 ):
            raise ValueError( "Incorrect dimensions for conditional/marginal probability" );
        
        if ( len( shape ) != len( dependencyList ) + 1 ):
            raise ValueError( "Incorrect dependency list" );
        
        u = self.vertexObject( uName );
        u.probability = probability;
        u.states = states;
        u.dependencyList = dependencyList;
    
    def jointProbability( self, vertexList = None ):
        '''
        Computes the joint probability of the network. If a vertex list is provided, the joint
        probability is computed only for the given nodes. If the evidence in the network is set, 
        the vertex list is ignored, and the joint probability of the entire network is computed.
        The corresponding dimensions of the joint probability are also collapsed when evidence is
        set on them.
        
        @param vertexList list of _Vertex objects to compute the joint probability. If None, all 
               vertices in the network are used
        '''
        if ( vertexList is None or self.evidenceList ):
            vertexList = self.vertexes( );
        
        probList = tuple( [ v.probability for v in vertexList ] );
        indexDict = { v.name: chr( i + 97 ) for i, v in enumerate( self.vertexes( ) ) };
        
        rhs = ''.join( [ indexDict[ v.name ] for v in vertexList ] );
        lhsList = [ ];
        
        for u in vertexList:
            lhsTerm = [ indexDict[ u.name ] ] + [ indexDict[ v ] for v in u.dependencyList ];
            lhsList.append( ''.join( lhsTerm ) );
        
        lhs = ','.join( lhsList );
        
        expr = '->'.join( [ lhs, rhs ] );
        jointProb =  np.einsum( expr, *probList );
        jointProb = jointProb / np.sum( jointProb );
        
        if ( not self.evidenceList ):
            return jointProb;
        else:
            dimList = [ slice( None ) ] * len( np.shape( jointProb ) );
            
            for vName, vState in self.evidenceList:
                vIndex = ord( indexDict[ vName ] ) - 97;
                dimList[ vIndex ] = self.vertexObject( vName ).states.index( vState );
            
            slicedProb = jointProb[ tuple( dimList ) ];
            slicedProb = slicedProb / np.sum( slicedProb );
            return slicedProb;
    
    def marginalProbability( self, uName, total = True ):
        '''
        Computes the marginal probability for a specified node using the joint probability. If total
        is false, only the joint probability of the node and its parents are used to compute the
        marginal. If it is true, the joint probability of the full network is used. If evidence is
        set, total is set to true regardless of its specified value.
        
        @param uName Name of the node to compute the marginal probability
        @param total If True, specifies that the joint probability of the complete network must be
               used to compute the marginal. If evidence is set, total is set to True regardless
               of input
        '''
        u = self.vertexObject( uName );
        
        if ( not u.dependencyList and not self.evidenceList ):
            return u.probability;
            
        if ( self.evidenceList ):
            total = True;
        
        if ( total ):
            vertexList = self.vertexes( );
        else:
            vertexList = [ u ] + [ self.vertexObject( vName ) for vName in u.dependencyList ];
        
        jointProb = self.jointProbability( vertexList );
        
        indexDict = { v.name: chr( i + 97 ) for i, v in enumerate( self.vertexes( ) ) };
        setVertexes = [ vName for vName, vState in self.evidenceList ];
        
        lhs = ''.join( [ indexDict[ v.name ] for v in vertexList if v.name not in setVertexes ] );
        rhs = indexDict[ uName ];
        
        expr = '->'.join( [ lhs, rhs ] );
        return np.einsum( expr, jointProb );
    
    def setEvidence( self, evidenceList ):
        '''
        Sets evidence in the network. The evidence list is provided as input
        
        @param evdienceList A list of tuple pairs containing the node and its state value to be set.
        '''
        for vName, vState in evidenceList:
            try:
                self.vertexObject( vName );
            except KeyError:
                print( "Invalid node specified" );
                return;
            
            if ( vState not in self.vertexObject( vName ).states ):
                raise ValueError( "Invalid state for node specified" );
            
        self.evidenceList = evidenceList;
    
    def unsetEvidence( self ):
        '''
        Unsets the set evidence on the network
        '''
        self.evidenceList = [ ];
    
    def getInference( self, uName ):
		'''
		Computes the marginal probability for a specified node and returns it
		
		@param uName The name of the node for which the inference is to be computed
		@return The inference on the given node for the network in the current state
		'''
        for vName, vState in self.evidenceList:
            if ( uName == vName ):
                return vState;
        
        return self.marginalProbability( uName, total = True );
    
    def __repr__( self ):
        return 'BayesianNet: [{}]'.format( ', '
                .join( [ str( k ) for k in sorted( self._edges.values( ) ) ] ) );



