'''
Created on Apr 19, 2015

@author: jorge

Creates classes DirectedGraph and UndirectedGraph
and private classes _Vertex and _Edge that users should not need to use by name
'''

import functools

@functools.total_ordering
class _Vertex(object):
    '''
    A Vertex in a graph. End users should not create their own Vertex objects. 
    They should call the function graph.addVertex instead.
    They can access the name and neighbor properties (but should not modify them) 
    and can add their own properties
    '''
    def __init__(self, name):
        '''
        Create a new Vertex
        @param name of the vertex 
        '''
        self.name = name
        self.neighbors = set() # the Vertexes that this Vertex is connected to via Edges

    '''
    Vertexes are compared according to their names
    '''
    def __hash__(self):
        return hash(self.name)
        
    def __eq__(self,other):
        return self.name == other.name        
        
    def __lt__(self,other):
        return self.name < other.name
    
    def __repr__(self):
        return str(self.name)
    
@functools.total_ordering
class _Edge(object):
    '''
    An Edge in a graph. End users should not create their own Edge objects. 
    They should call the function graph.addEdge instead.
    They can access the p,c and weight properties (but should not modify them) 
    and can add their own properties
    '''
    def __init__(self, p, c, weight=1):
        '''
        Create a new Edge where:
        @param p is the 'parent' Vertex in case of a Directed Edge, or the smaller Vertex in case of an Undirected Edge
        @param c is the 'child' Vertex in case of a Directed Edge, or the larger Vertex in case of an Undirected Edge
        @param weight is the 'weight' of the Edge    
        '''
        self.p = p           
        self.c = c          
        self.weight = weight
        
    '''
    Edges are compared according to the tuple (p,c)
    '''
    def __hash__(self):
        return hash((self.p,self.c)) 
        
    def __eq__(self,other):
        return (self.p,self.c) == (other.p,other.c) 
                  
    def __lt__(self,other):
        return (self.p,self.c) < (other.p,other.c) 
        
    def __repr__(self):
        return str((self.p,self.c))
    
class DirectedGraph(object):
    '''
    A Directed Graph
    Edge ends are represented as (parent, child)
    A child Vertex is a neighbor of a parent Vertex, but not viceversa
    '''
    
    def addVertex(self,name):
        '''
        Add a vertex with a given name to this graph if it did not already exist
        @param name of the vertex to add
        @return the _Vertex object that name refers to
        '''
        if name not in self._vertexes:
            self._vertexes[name] = _Vertex(name)
        return self._vertexes[name]
            
    def addEdge(self,pName,cName,weight=1):
        '''
        Add an edge to this graph if it did not already exist between the vertexes named
        pName and cName. 
        Add those vertexes p and c if they did not exist
        Add c as a neighbor of p
        @param pName is the 'parent' Vertex
        @param cName is the 'child' Vertex
        @param weight is the 'weight' of the Edge
        @return the Edge object between p and c
        '''
        if (pName,cName) not in self._edges:
            p = self.addVertex(pName) # addVertex is idempotent!
            c = self.addVertex(cName)
            p.neighbors.add(c)
            self._edges[(pName,cName)] = _Edge(p,c, weight)
        return self._edges[(pName,cName)]
    
    def vertexes(self):
        '''
        @return the set of Vertexes in the graph. Users can add their own properties to them
        '''
        return self._vertexes.values()
    
    def vertexObject( self, vName ):
        '''
        @param  vName is the vertex name
        @return the vertex object corresponding to the vertex name vname
        '''
        return self._vertexes[ vName ];

    def edges(self):
        '''
        @return the set of Edges in the graph. Users can add their own properties to them
        '''
        return self._edges.values() 

    def __init__(self, vertexes = [], edges = []):
        self._vertexes = {}
        for v in vertexes:
            self.addVertex(v)
             
        self._edges = {}
        for e in edges:
            self.addEdge(*e)
            
    def __repr__(self):
        return 'DirectedGraph: [{}]'.format(', '.join([str(k) for k in sorted(self._edges.items())]))
    
class UndirectedGraph(DirectedGraph):
    '''
    An Undirected Graph.
    Only ene Edge per pair of nodes representing both directions. Edge uses canonical representation with their end names sorted. 
    Both end Vertexes of an Edge are neighbors of each other.
    '''
            
    def addEdge(self,uName,vName,weight=1):
        '''
        Add an edge to this graph if it did not already exist.
        
        @param uName is one end point Vertex of this Edge
        @param vName is the other end point Vertex of this Edge
        @param weight is the 'weight' of the Edge
        @return the Edge object between u and v
        
        We remark that in Undirected graphs, the Vertexes u and v are fully symmetric. (That is, the 
        edge (u,v) is the same as the edge (v,u). By convention they are represented canonically such that u<v,
        Thus this function calls the DirectedGraph version of addEdge using the canonical order and in addition 
        adds u as a neighbor of v.
        '''
        uName,vName = (uName,vName) if uName<vName else (vName,uName) # canonical orientation for the edge
        edge = super().addEdge(uName,vName,weight) 
        self._vertexes[vName].neighbors.add(self._vertexes[uName])
        return edge
    
    def __repr__(self):
        return 'UndirectedGraph: [{}]'.format(', '.join([str(k) for k in sorted(self._edges.values())]))


