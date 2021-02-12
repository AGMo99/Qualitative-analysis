#Load Packages

using CSV, DataFrames, CategoricalArrays, HypothesisTests, Statistics, StatsBase, GLM, Gadfly, DataStructures,
     Distributions,LinearAlgebra

#Load Data

df = DataFrame(CSV.File("C:/Users/AG/desktop/data/Micro data.csv", header = true))

#Clean
df = df[df[:, 1].== 2020,:]
df = df[df[:, 2].=="price_expect_own_rent", :]

select!(df, Not(:Year))
select!(df, Not(:category))
select!(df, Not(2:23))
select!(df, Not(8:65))
select!(df, Not(:obs))

#Change Column's Name

df = select(df,"demographics"=> "demographics", "Verybadinvestment" => "Very Bad",
     "Somewhatbadinvestment" => "Some Bad","Neithergoodnorbadasaninvestment"  => "Neither",
	 "Somewhatgoodinvestment" => "Some Good", "Verygoodinvestment" => "Very Good")

DataFrames.describe(df)

DataFrames.head(df)

#Categorical Column
unam = ["own", "rent"]
uval = [1, 0]

y = DataFrame(demographics = unam, ind = uval)

df = join(df, y, on = "demographics")

select!(df, Not(:demographics))

#Change Form % to count number

df["Very Bad"][df["Very Bad"] .==[2, 1]] = [15, 2]
df["Some Bad"][df["Some Bad"] .==[7, 6]] = [52, 15]
df["Neither"][df["Neither"] .==[19, 25]] = [142, 62]
df["Some Good"][df["Some Good"] .==[48, 41]] = [58, 102]
df["Very Good"][df["Very Good"] .==[24, 27]] = [179, 67]

#Plot The Data

plot(df, color="ind", x="Very Good", y="Very Bad", shape="Neither", Geom.point)
plot(df, color="ind", x="Some Good", y="Some Bad", Geom.point)

#The Model

model = GLM.glm(@formula(ind ~ Very Bad + Some Bad + Neither + Some Good + Very Good), df, Nromal(), ProbitLink())

res = df.ind - predict(model)

#Check The Model

JarqueBeraTest(predict(model))

DurbinWatsonTest(df.ind, res)
