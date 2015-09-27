# -*- coding: utf-8 -*-
"""
Created on Tue May 19 11:51:49 2015

@author: Ameer Asif Khan
"""
import numpy as np;
import os;

os.chdir( "C:/Users/Ameer Asif Khan/My Documents/Academic/Northwestern/" + 
            "MSIA 490 Topics in Analytics in Python" );

import bayesian;

#%%
a = bayesian.BayesianNet( );
a.addNode( 'Smoking' );
a.addChild( 'Smoking', 'Cancer' );

a.addProbabilityTable( 'Smoking', np.array( [ 0.8, 0.15, 0.05 ] ),
                       [ 'None', 'Light', 'Heavy' ] );
a.addProbabilityTable( 'Cancer', np.array( [ [ 0.96, 0.88, 0.60 ],
                                             [ 0.03, 0.08, 0.25 ],
                                             [ 0.01, 0.04, 0.15 ] ] ),
                       [ 'None', 'Benign', 'Malignant' ], [ 'Smoking' ] );
prob = a.jointProbability( );

a.setEvidence( [ ( 'Smoking', 'Light' ) ] );

#%%
b = bayesian.BayesianNet( );
b.addNode( 'cloudy' );
b.addChild( 'cloudy', 'sprinkler' );
b.addChild( 'cloudy', 'rain' );
b.addChild( 'sprinkler', 'grass wet' );
b.addChild( 'rain', 'grass wet' );

b.addProbabilityTable( 'cloudy', np.array( [ 0.5, 0.5 ] ), [ 'T', 'F' ] );
b.addProbabilityTable( 'sprinkler', np.array( [ [ 0.1, 0.5 ],
                                                [ 0.9, 0.5 ] ] ),
                       [ 'T', 'F' ], [ 'cloudy' ] );
b.addProbabilityTable( 'rain', np.array( [ [ 0.8, 0.2 ],
                                           [ 0.2, 0.8 ] ] ),
                       [ 'T', 'F' ], [ 'cloudy' ] );
b.addProbabilityTable( 'grass wet', np.array( [ [ [ 0.99, 0.9 ],
                                                  [ 0.9, 0 ] ],
                                                [ [ 0.01, 0.1 ],
                                                  [ 0.1, 1 ] ] ] ),
                       [ 'T', 'F' ], [ 'sprinkler', 'rain' ] );

prob = b.jointProbability( );
b.marginalProbability( 'sprinkler' );
b.setEvidence( [ ( 'rain', 'T' ) ] );
b.getInference( 'grass wet' );
b.unsetEvidence( );


