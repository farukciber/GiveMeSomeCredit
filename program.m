%TRENIRANJE
clear all; clc;

%Citanje fajla
trainData = csvread('cs-training.csv' , 1 , 1);

%Rasporedjivanje varijabli
train_X = trainData(:, 2:11);
Y = trainData(:, 1);

%Kolona 6 i 11 u excelu imaju neke vrijednosti NaN i njih moramo popuniti sa
%prosjekom te kolone

%Nadjemo na kojem indexu je NaN i na kojem nije Nan
nanIndices5 = find(all(isnan(train_X(:,5)),2));
notNanIndices5 = find(all(~isnan(train_X(:,5)),2));

%Ovdje popunimo NaN vrijednosti sa prosjekom iz te kolone
train_X(nanIndices5, 5) = sum(train_X(notNanIndices5, 5)) / length(notNanIndices5);

%Nadjemo na kojem indexu je NaN i na kojem nije Nan
nanIndices10 = find(all(isnan(train_X(:,10)),2));
notNanIndices10 = find(all(~isnan(train_X(:,10)),2));

%Ovdje popunimo NaN vrijednosti sa prosjekom iz te kolone
train_X(nanIndices10 , 10) = sum(train_X(notNanIndices10 , 10)) / length(notNanIndices10);

%Mean normalization
train_X = featureScaling(train_X);

%Izracunamo broj redova i kolona
[numRows, numColumns] = size(train_X);

%Dodamo broj jedinica radi dimenzija
train_X = [ones(numRows, 1) train_X];

%Odredimo thete
thetas = zeros(numColumns + 1, 1);

%Za regularizaciju
lambda = 1;

%Neke optimizacije sa Coursere
options = optimset('GradObj', 'on', 'MaxIter', 500);
[theta, costFunctionRegularization, exit_flag] = fminunc(@(t)(costFunction(t, train_X, Y, lambda)), thetas, options);
	
%Predictamo 0 ili 1 ovisno da li je >=0.5
p = (sigmoid(train_X*theta) >= 0.5);

%Trazimo indexe gdje su p i y jednaki
sum1 = 0;
for i=1:length(Y)
    if(p(i)==Y(i))
        sum1 = sum1 + 1;
    end
end

%Izracunamo prosjek koliko je jednako p i y
average = sum1/length(Y)*100;

%Ispis predikcije
fprintf('Prediction: %f', average);




%TESTIRANJE

%Citanje fajla
test_X = csvread('cs-test.csv' , 1 , 2);

%Broj redova i kolona
[testRow , testCol] = size(test_X);

%Kolona 6 i 11 u excelu imaju neke vrijednosti NaN i njih moramo popuniti sa
%prosjekom te kolone

%Nadjemo na kojem indexu je NaN i na kojem nije Nan
nanIndices5 = find(all(isnan(test_X(:,5)),2));
notNanIndices5 = find(all(~isnan(test_X(:,5)),2));

%Ovdje popunimo NaN vrijednosti sa prosjekom iz te kolone
test_X(nanIndices5, 5) = 1.0 * sum(test_X(notNanIndices5, 5)) / length(notNanIndices5);

%Nadjemo na kojem indexu je NaN i na kojem nije Nan
nanIndices10 = find(all(isnan(test_X(:,10)),2));
notNanIndices10 = find(all(~isnan(test_X(:,10)),2));

%Ovdje popunimo NaN vrijednosti sa prosjekom iz te kolone
test_X(nanIndices10 , 10) = sum(test_X(notNanIndices10 , 10)) / length(notNanIndices10);


%Mean normalizacija
test_X = featureScaling(test_X);

%Dodati redove jedinica
test_X = [ones(testRow , 1) test_X];

%Predikcija od 0 do 1 da li ce osoba imati distress
prediction = sigmoid(test_X * theta);

%Matrica za svaku osobu
indices = (1:testRow)';

%Pisanje u fajl za svaku osobu predikciju
dlmwrite('C:\Users\Faruk\Desktop\Logistic Regression 301015037\predictions.csv' , [indices , prediction]);


