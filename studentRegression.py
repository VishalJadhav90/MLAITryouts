def studentReg(ages_train, net_worths_train):
    ### import the sklearn regression module, create, and train your regression
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression()
    ### name your regression reg

    ### your code goes here!
    reg.fit(ages_train, net_worths_train)
    return reg
