%% Preprocessing data

train = readtable('train.csv');

% We get rid of the "uninformative" data
train = removevars(train,{'PassengerId','Ticket','Cabin'}); 

% Categorizing categorical data
train.Survived = categorical(train.Survived);
valueset = [3:-1:1];
catnames = {'Third class' 'Second class' 'First class'};
train.Pclass = categorical(train.Pclass, valueset, catnames, 'Ordinal', true);
clear valueset catnames
train.Sex = categorical(train.Sex);
train.Embarked = categorical(train.Embarked);

% Processing the names into classes
for ii = 1:length(train.Name)
    firstwords = extractBefore(train.Name(ii),".");
    splitted = split(firstwords,', ');
    train.Name(ii) = splitted(end);
end
clear firstwords splitted ii
train.Name = categorical(train.Name);
train.Name = mergecats(train.Name,{'Capt' 'Col' 'Don' 'Dr' 'Jonkheer' 'Lady' 'Major' 'Mlle' 'Mme' 'Ms' 'Rev' 'Sir' 'the Countess' 'Master'},'Other');

X = train(:,2:end);
Y = table2array(train(:,1));
% Y = ind2vec(Y')';

% We train the model using the Classification Learner

%% Preprocessing test data

test = readtable('test.csv');

% We get rid of the "uninformative" data
PassengerId = test(:,1);
test = removevars(test,{'PassengerId','Ticket','Cabin'}); 

% Categorizing categorical data
valueset = [3:-1:1];
catnames = {'Third class' 'Second class' 'First class'};
test.Pclass = categorical(test.Pclass, valueset, catnames, 'Ordinal', true);
clear valueset catnames
test.Sex = categorical(test.Sex);
test.Embarked = categorical(test.Embarked);

% Processing the names into classes
for ii = 1:length(test.Name)
    firstwords = extractBefore(test.Name(ii),".");
    splitted = split(firstwords,', ');
    test.Name(ii) = splitted(end);
end
clear firstwords splitted ii
test.Name = categorical(test.Name);
test.Name = mergecats(test.Name,{'Capt' 'Col' 'Don' 'Dr' 'Jonkheer' 'Lady' 'Major' 'Mlle' 'Mme' 'Ms' 'Rev' 'Sir' 'the Countess' 'Master'},'Other');

%% Finally we use the model

load('trainedModel.mat')
Survived = trainedModel.predictFcn(test);
writetable([PassengerId table(Survived)],'Yfit.csv')