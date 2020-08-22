# Recomendation-Systems

A Recommender System refers to a filtering procedure that is capable of predicting the most liked preference of a set of items  for a user.

Lets first talk about types of data in recommendation systems.

There are 2 types of data - 
1.Explicit data - Basically, it is the information provided by user itself, 
2.Implicit data - It is the data that is tracked about the user's likes and dislikes basing upon the websites he opens, products he purchases and so on..

Examples of explicit data collection include the following:
  Asking a user to rate an item on a sliding scale,
  Asking a user to search,
  Asking a user to rank a collection of items from favorite to least favorite,
  Presenting two items to a user and asking him/her to choose the better one of them,
  Asking a user to create a list of items that he/she likes.
  
Examples of implicit data collection include the following:
  Observing the items that a user views in an online store,
  Analyzing item/user viewing times,
  Keeping a record of the items that a user purchases online,
  Obtaining a list of items that a user has listened to or watched on his/her computer,
  Analyzing the user's social network and discovering similar likes and dislikes.
  
Few important Terminologies used in this context: 

ITEM / DOCUMENTS - Entities which are to be recommended. Eg. apps in google play store , movies in netflix , products to buy in amazon shopping app.

USER - Customer or member using items.

QUERY / CONTEXT - Information about user.  (explicit and implicit information about user)

EMBEDDING SPACE(Dimensionality) - Mapping from a set of queries/ items to a vector space.  


BASIC ARCHITECTURE OF RECOMMENDATION SYSTEM ( 3 STAGE ) (check out this for more details-https://developers.google.com/machine-learning/recommendation )

 - CANDIDATE GENERATION - FIRST STAGE OF POTENTIALLY REDUCING FROM A LARGER DATA TO SMALLER SUBSET.  

 - SCORING - SECOND STAGE OF SCALING DOWN THE OBTAINED SUBSET IN CANDIDATE GENERATION TO A PRECISELY SMALLER SET OF ITEMS TO BE RECOMMENDED TO THE USER

 - RE-RANKING - THIS IS FURTHER STAGE WHERE THE RECOMMENDED SET FROM SCORING IS FILTERED PROPERLY REMOVING EXPLICITLY DISLIKED ITEMS BY USER.

OVERVIEW OF CANDIDATE GENERATION:

 1.CONTENT BASED RECOMMENDATION SYSTEM
(APPROACHES - REGRESSION AND CLASSIFICATION OR EVEN USING NLP)

 2.COLLABORATIVE FILTERING RECOMMENDATION SYSTEM
(APPROACHES - 1. NEAREST NEIGHBORS , 2. LATENT FACTORS METHOD( MATRIX FACTORIZATION , DEEP LEARNING TECHNIQUE ))

 3.HYBRID TECHNIQUE 


CONTENT BASED RECOMMENDATION SYSTEM:
  It recommends items to users based on explicit information provided by user.In content based approaches, the recommendation problem is either casted into classification problem (predict if user likes “i” item or not) or into regression problem ( predict the rating given by user to an item “i”).If our classification(or regression) is based on user features, then it is called ITEM CENTERED and if we are working with item features, then it is called USER CENTERED. 
 - ITEM CENTERED : what is the probability of each user to like this item ( or what is the rate given by each user to this item ).

 - USER CENTERED : what is the probability of a particular user to like each item ( or what is the rate given by a user to each item ).

In most of the cases , the user may not want to answer too many questions. So, let's discuss few convenient approaches of content based filtering method.
 1.Item-centered Bayesian classifier ( INPUT - USER FEATURES
OUTPUT - LIKE OR DISLIKE ITEM )

 2.User-centered Linear Regression. ( INPUT - ITEM FEATURES
OUTPUT - PREDICT RATING )

There are many other more accurate classification or regression ML models which can be used for item-centered classification or user-centered regression. 

ITEM - CENTERED
Suppose,
PITEM ( LIKE | USER_FEATURES ) = ( PITEM ( USER_FEATURES | LIKE ) * PITEM ( LIKE) ) / ( PITEM ( USER_FEATURES ) )
PITEM ( DISLIKE | USER_FEATURES ) =  ( PITEM ( USER_FEATURES | DISLIKE ) * PITEM ( DISLIKE) ) /  PITEM ( USER_FEATURES ) )
PITEM ( LIKE )  and  PITEM (  DISLIKE  )  are probabilities of item being liked and disliked by the user 
PITEM ( . | LIKE )  and PITEM ( . | DISLIKE ) are likelihoods which are assumed to follow Normal Distributions.

We have to compute           PITEM ( LIKE | USER_FEATURES ) / PITEM (USER_FEATURES ) , This probablity tells how much a particular item is liked by the user.


PROS AND CONS OF CONTENT BASED FILTERING METHOD. 

 - PROS :

THE MODEL DOESN’T NEED DATA ABOUT ANY OTHER USERS.
THE RECOMMENDATIONS ARE SPECIFIC TO THIS USER. 

 - CONS:

IF THE CONTENT DOESN’T CONTAIN ENOUGH INFORMATION TO DISCRIMINATE THE ITEMS PRECISELY, THE RECOMMENDATION WILL NOT BE PRECISE AT THE END.
  THE RECOMMENDATION CANNOT BE PROVIDED CORRECTLY FOR NEW USER AS THERE IS NO PROPER PROFILE.


COLLABORATIVE FILTERING BASED RECOMMENDATION SYSTEM

A COLLABORATIVE TYPE SYSTEM RECOMMENDS ITEMS TO USER A, BASED ON HISTORIC USER A PREFERENCES FOR ITEMS(CLICKED, WATCHED , PURCHASED , LIKED) WHICH MATCHED WITH PREFERENCES OF USER B.THIS METHOD OF RECOMMENDATION IS BASED ON USER BEHAVIOR TO DETECT SIMILAR USERS/ITEMS TO RECOMMEND ITEMS.THE RECOMMENDATION SYSTEM IS FURTHER CLASSIFIED AS MEMORY BASED AND MODEL BASED.

Memory based method do not assume any latent model . The algorithm works directly on the user-item interaction. Thus, it has low bias and high variance.
Eg :- Nearest neighbors work on these interactions to produce suggestions.

Model based methods assumes latent interaction model , having a mathematical meaning. It has high bias and low variance.
Content based method also has latent interaction model built around users and items explicitly given. Thus it has highest bias and lowest variance.


Memory based collaborative approach:

 let's say you have a given user "A" and a given item "I" but the user hasn't rated the item, you want to estimate that rating.
There are two basic approaches to memory based collaborative filtering:

 - User-User similarity: (“people like you like that” logic):  You find users (using nearest neighbors) that are similar to "A" and then you estimate the rating of "I" based on what those users think about the item.

 - Item-Item similarity: (“if you like this you might also like that” logic):  You find items (using nearest neighbors) that are similar to "I" and then you estimate the rating of I based on what "A" thinks about those items.

Model based collaborative approach:

This approach relies on latent based user-item interaction matrix.
The interaction between an item and user is computed by dot product of corresponding dense vector in the embedded space.
This method tries to reduce dimensionality of the interaction matrix and approximate it by two or more matrices with k latent components.

COLLABORATIVE FILTERING APPROACHES:
1.NEAREST NEIGHBORS 
2.LATENT FACTORS
 - MATRIX FACTORIZATION
 - DEEP LEARNING TECHNIQUE

LATENT FACTORS
MATHEMATICS OF MATRIX FACTORIZATION 
Consider an interaction matrix M (n*m) of ratings where only some are explicitly given by user. We want to factorize matrix M as 
M = X.YT   
X = user matrix of n users 
Y= item matrix of m items
Here , l is the dimension of latent space.

CHOOSING OBJECTIVE FUNCTION
1.We have to minimize the rating reconstruction error which is given by 
  (X,Y) = argminX,Y sumof[(Xi)(Yj)^T - Mij]^2 where i,j belongs to belongs to E.
  This is also known as OBSERVED ONLY MATRIX FACTORIZATION.
  
2.If sum over unobserved/ not rated entries are considered as zeros , then it is WEIGHTED MATRIX FACTORIZATION, where w0 is the hyperparameter.
  (X,Y) = argminX,Y sumof[(Xi)(Yj)^T - Mij]^2 (where i,j belongs to belongs to E.
)+ w0 sumof[(Xi)(Yj)^T - 0]^2 (where i,j doesnt belongs to belongs to E).

3 SINGULAR VALUE DECOMPOSITION
If all non rated values are given as zeroes and try to minimize sum of all entries in matrix , then it is called minimizing the squared frobenius distance between M and    X.Y^T   
(X,Y) = argminX,Y sumof[(Xi)(Yj)^T - Mij]^2 for all i,j 

This quadratic equation is solved using SVD.  The goal of using this is to find latent factors and also reduce the size of dimensionality.

Common algorithms to minimize the objective function are:
1.Stochastic Gradient Descent (SGD) - Updates each parameter independently.
2.Weighted Alternating Least Squares ( WALS )- Can be parallelized.
(Fixes X and solves for Y.  &  Fixes Y and solves for X.
Each step guarantees in decrease in loss).

PROS AND CONS OF COLLABORATIVE FILTERING

 - PROS

SERENDIPITY - the model helps users to discover new interests.

 - CONS

COLD START PROBLEM - 
Accurate recommendations cannot be made for new users/items with no or little information.
There are two ways to overcome cold start problem ,
Projection in WALS,
Heuristics to generate embeddings of fresh item.




SIMILARITY MEASURES (DEGREE OF CLOSENESS)

Let S( q, x) be the similarity function. Where q is query(or user) embedding of the user and x  is item embedding. We need to recommend item embedding which are close to q embedding. The degree of similarity is measured in three different ways.
 1.COSINE RULE - Cos (q,x) ; More the value , higher the similarity.
 2.DOT PRODUCT - |q| |x| Cos(q,x) ; Upon normalization, Dot product similarity becomes equal to Cosine rule. More the value of dot product , higher is the similarity .
 3.EUCLIDEAN DISTANCE - sqrt(  (|q| - |x| )^(2) ) ; Smaller the distance, higher the similarity.


