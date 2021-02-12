#Load Packages

using CSV, DataFrames, CategoricalArrays, HypothesisTests, Statistics, StatsBase, GLM, Gadfly, DataStructures,
     Distributions,LinearAlgebra

#Load Data

df = DataFrame(CSV.File(".../Micro data.csv", header = true))

#Clean
df = df[df[:, 1].== 2020,:]
df = df[df[:, 2].=="price_expect_own_rent", :]

select!(df, Not(:Year))
select!(df, Not(:category))
select!(df, Not(2:23))
select!(df, Not(8:65))
select!(df, Not(:obs))

#Change Column's Name

df = select(df,"demographics"=> "demographics", "Verybadinvestment" => "VeryBad",
     "Somewhatbadinvestment" => "SomeBad","Neithergoodnorbadasaninvestment"  => "Neither",
	 "Somewhatgoodinvestment" => "SomeGood", "Verygoodinvestment" => "VeryGood")

DataFrames.describe(df)

DataFrames.head(df)

#Categorical Column
unam = ["own", "rent"]
uval = [1, 0]

y = DataFrame(demographics = unam, ind = uval)

df = join(df, y, on = "demographics")

select!(df, Not(:demographics))

#Change Form % to count number

df["VeryBad"][df["VeryBad"] .==[2, 1]] = [15, 2]
df["SomeBad"][df["SomeBad"] .==[7, 6]] = [52, 15]
df["Neither"][df["Neither"] .==[19, 25]] = [142, 62]
df["SomeGood"][df["SomeGood"] .==[48, 41]] = [58, 102]
df["VeryGood"][df["VeryGood"] .==[24, 27]] = [179, 67]

#Plot The Data

plot(df, color="ind", x="VeryGood", y="VeryBad", shape="Neither", Geom.point)
plot(df, color="ind", x="SomeGood", y="SomeBad", Geom.point)

#The Model

using scikitlearn

@sk_import linear_model: LogisticRegression

model = LogisticRegression(fit_intercept=true)
fit!(model,ind, VeryBad, SomeBad, Neither, SomeGood, VeryGood)

ps = predicte_proba

res = df.ind - predict(model)

#Check The Model

JarqueBeraTest(predict(model))

DurbinWatsonTest(df.ind, res)
