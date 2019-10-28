// Databricks notebook source
// MAGIC %md ###My First Project (Studying Divorces & Marriages in Singapore)
// MAGIC #####Hello there! I've just recently learnt about Scala & Spark and I'm hoping to get some hands on practice on it. If you're new to it too, I hope this little project could play a part in your learning. So without further ado, lets dive in further into this project!

// COMMAND ----------

// MAGIC %md ######In this project, we will take a look into whether we can understand and analyze Singapore's key marriage and divorce data better. I specifically got these datasets from Singapore's own Open Data Portal (https://data.gov.sg/). The portal actually shares a lot of interesting & relevant data that we can play around and analyze with so if you're interested to find your own datasets you can take some time to search for them there during your own free time. So let me first begin by importing the necessary libraries and datasets into the notebook for me to begin.

// COMMAND ----------

// MAGIC %md ###Divorces in Singapore

// COMMAND ----------

import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.functions._

// COMMAND ----------

//We will study 3 different datasets on divorces in Singapore and see what findings we can achieve from them.
val totalDivorcesByDivorceType = spark.read.option("header", "true").option("inferSchema","true").csv("/FileStore/tables/total_divorces_by_divorce_type-8ec88.csv")
val medianAgeBySexDivorceType= spark.read.option("header", "true").option("inferSchema","true").csv("/FileStore/tables/median_age_by_sex_and_divorce_type_median_duration_of_marriage_by_divorce_type-22bd6.csv") 
val divorcesWomensCharter= spark.read.option("header", "true").option("inferSchema","true").csv("/FileStore/tables/DIVORC_2-97943.CSV") 

// COMMAND ----------

//medianAgeBySexDivorceType.orderBy(desc("year")).show(3)
//totalDivorcesByDivorceType.orderBy(desc("year")).show(3)
//divorcesWomensCharter.orderBy(desc("year")).show(2)
for(line <- medianAgeBySexDivorceType.head(5)){
  println(line)
}
println("\n")
for(line <- divorcesWomensCharter.orderBy(desc("year")).head(4)){
  println(line)
}
println("\n")
for(line <- totalDivorcesByDivorceType.orderBy(desc("year")).head(4)){
  println(line)
}

// COMMAND ----------

// MAGIC %md ######Alright! So we've finally imported the relevant datasets that we want to take a closer look at. Lets first study the Divorce data that we have to see what we can discover.

// COMMAND ----------

totalDivorcesByDivorceType.registerTempTable("totalDivorcesByDivorceType")
display(sqlContext.sql("select * from totalDivorcesByDivorceType where year > 1994"))


// COMMAND ----------

// MAGIC %md ######The first dataset on divorces contains data on the annual number of divorces under the Woman's Charter & The Administration of Muslim Law Act. A quick bar chart plot view shows that there has been a steady increase in the number of divorces under the Woman's Charter since 1995. In contrast, divorces under the Administration of Muslim Law Act also had a relative increase but the annual number of divorces under this has been relatively consistent year on year from 2005 onwards. This increase in divorce rates could possibly be an alarming issue but there's little data to support the reasons behind this. Could it be due to the cost of living or perhaps other social factors arising in Singapore? Perhaps a myriad of factors behind them. Nevertheless, let's continue looking at the other divorce related datasets that we have for now and see what we can uncover. Moving forward, lets specifically pay closer attention to divorces under the Woman's Charter since increase in divorce rates under that category is more prevalent among the two.

// COMMAND ----------

//The second dataset on divorce has a bit more data columns that we can play with. Namely the median age of males/females divorcees & the duration of marriage before the divorce itself. So lets break this dataset down so that we can take a closer look at both areas.
val medianAgeDivorceWomanCharterMalesAge = medianAgeBySexDivorceType.filter($"level_1" === "Median Age Of Male Divorcees")
 
medianAgeDivorceWomanCharterMalesAge.registerTempTable("medianAgeDivorceWomanCharterMalesAge")
display(sqlContext.sql("select * from medianAgeDivorceWomanCharterMalesAge where year > 1995"))


// COMMAND ----------

val medianAgeDivorceWomanCharterFemalesAge = medianAgeBySexDivorceType.filter($"level_1" === "Median Age Of Female Divorcees")
 
medianAgeDivorceWomanCharterFemalesAge.registerTempTable("medianAgeDivorceWomanCharterFemalesAge")
display(sqlContext.sql("select * from medianAgeDivorceWomanCharterFemalesAge where year > 1995"))


// COMMAND ----------

// MAGIC %md ######As we can see above, the median age of both males & females that experienced a divorce has increased year on year. Interesting to note because given that the age at which a divorce happens increases year on year, one might think that it could possibly be the exact same generation that gets older year on year that becomes unhappy and undergoes a divorce. But that's just a hypothesis and we would probably need more data to support such a statement. One thing for sure is that we can conclude that the median age of males/females who divorce is likely to increase year on year. Just to be sure lets create and take a look at the line graph comparing the two gender's age below.

// COMMAND ----------

val medianAgeDivorceWomanCharter = medianAgeBySexDivorceType.filter($"level_2" === "Under The Women'S Charter")//Filter to just Womens Charter   
medianAgeDivorceWomanCharter.registerTempTable("medianAgeDivorceWomanCharter")
display(sqlContext.sql("select * from medianAgeDivorceWomanCharter where level_1 <> 'Median Duration Of Marriage For Divorces'"))

// COMMAND ----------

// MAGIC %md ######As expected, their median ages have increased for both genders year on year. Lets dive in further take a look at the other aspect of this dataset, duration of marriage, to see if this has any impact or relevance on divorce rates here in Singapore.

// COMMAND ----------

display(sqlContext.sql("select * from medianAgeDivorceWomanCharter where level_1 <> 'Median Age Of Male Divorcees' and level_1 <> 'Median Age Of Female Divorcees'"))

// COMMAND ----------

// MAGIC %md ######A quick analysis on the median duration of marriage before divorce above shows that majority of the divorces in the median range last for at most 10-11 years. Perhaps this data may be useful for us to predict the divorce rates in the future. I'm still new to Scala & ML but I would definitely be exploring this more in the future. Let's leave this here for now & move to our final dataset on divorce. The last dataset on divorces focuses more on divorces under the Women's Charter and has an interesting data pertaining to the previous marital status of both spouses before their divorce itself. This may have particularly interesting information that could help us determine if their previous marital statuses have any impact whatsoever on their latest divorce.

// COMMAND ----------

import org.apache.spark.sql.functions.countDistinct
divorcesWomensCharter.describe().show()
divorcesWomensCharter.agg(countDistinct("level_2")).show()

// COMMAND ----------

println("\n")
for(line <- divorcesWomensCharter.orderBy(desc("year")).head(9)){
  println(line)
}

// COMMAND ----------

// MAGIC %md ######This next data set contains data related to the marital status of each spouse before they married and divorced. A quick check shows that we have a total of 9 types of possible couple combinations that divorced. Theres plenty of room to analyse this set of data. So for now, I'll focus on singling out the data on "Single Man & Divorced Woman" & "Divorced Man & Single Woman" and see if theres any stark differences between them. 

// COMMAND ----------

val df = divorcesWomensCharter.filter($"level_2" === "Divorced Man & Single Woman" || $"level_2"==="Single Man & Divorced Woman")
val dfByRace = df.filter($"level_1" =!= "Total")
val dfByTotal = df.filter($"level_1" === "Total")
println("\n")
for(line <- dfByRace.head(7)){
  println(line)
}
println("\n")
for(line <- dfByTotal.head(7)){
  println(line)
}

// COMMAND ----------

dfByTotal.registerTempTable("dfByTotal")
display(sqlContext.sql("select * from dfByTotal where year > 1995"))

// COMMAND ----------

// MAGIC %md ###### As we can see, number of divorces for these 2 groups have been increasing across the last 20 years. An interesting anamoly did happen in 2004 where there was an exceptionally lower number of divorces for couples that consisted of a single & divorced counterpart in their previous relationship which makes me curious why this is so. Another observation to take away from this is that couples who had a divorced man as a partner seemed to be more likely to experience a divorce in comparison to couples that involved a divorced woman. That doesn't sound too great for divorced men looking to succeed in their subsequent marriages. :( Since we have data related to their etnicity lets see why we can find related to that.  

// COMMAND ----------

dfByRace.registerTempTable("dfByRace")
display(sqlContext.sql("select * from dfByRace where year == 2018"))

// COMMAND ----------

display(sqlContext.sql("select * from dfByRace where year > 1990"))

// COMMAND ----------

// MAGIC %md ###### A quick look for the year 2018 showed that inter ethnic marriages where the man was previously divorced and the woman was single seemed to be less likely to succeed and ended in a divorce. The total number of interethnic marriages with a "Divorced Man & Single Woman" was higher at 63% in comparison to "Single Man & Divorced Woman" at 37%. Its interesting to note that for Indians, it was the other way round in comparison to the other ethnicities displayed in which there was more divorces for "Single Man & Divorced Woman" than for "Divorced Man & Single Women" with a 10% difference.

// COMMAND ----------

val dfNewTotal = divorcesWomensCharter.filter($"level_1" === "Inter-ethnic")
dfNewTotal.registerTempTable("dfNewTotal")
display(sqlContext.sql("select * from dfNewTotal where year > 1990"))

// COMMAND ----------

// MAGIC %md ######If we actually look specifically at inter-ethnic marriagges you can see that the number of divorces related to interethic couples have been on the rise as well. So given that there has been more interethnic related divorces, one would think that more resources should be focused on developing counselling programmes that can cater to inter-ethnic relationships. Different ethnic groups can have various cultures and practices especially when it comes to trying to piece back together a broken marriage or simply having to go through a divorce more smoothly. This growing trend of interethnic divorces would mean that more resources should be planned to help this group of people.

// COMMAND ----------

// MAGIC %md ###Marriages in Singapore

// COMMAND ----------

// MAGIC %md ######Enough of Divorces for now. Lets take a look at data related to Marriages in Singapore. In particular, lets look at 2 different datasets. The first data set consists of data on marriages and age differences between the groom and bride. The second data looks at the marriages by order, whether it is a couple's first marriage or a remarriage across the years.

// COMMAND ----------

val totalMarriages = spark.read.option("header", "true").option("inferSchema","true").csv("/FileStore/tables/total_marriages_by_marriage_order_and_age_differential_of_grooms_to_brides-0beae.csv") 
val totalMarriagesOrder = spark.read.option("header", "true").option("inferSchema","true").csv("/FileStore/tables/total_marriages_by_marriage_order-a8d8d.csv") 


// COMMAND ----------

println("\n")
for(line <- totalMarriages.head(6)){
  println(line)
}
println("\n")
for(line <- totalMarriagesOrder.head(6)){
  println(line)
}

// COMMAND ----------

totalMarriages.describe().show()

// COMMAND ----------

// MAGIC %md ######Since I'm only interested to look at the total marriages (First or Remarried), let me just remove the irrelevant columns first so that we can narrow down on those with Total marriages in the data. And since theres a lot of different age gaps, lets just pick a few to take a look at (Same Age, Younger By 2 Years, Older by 5 Years). Lets create a line plot to visualise this.

// COMMAND ----------

val marriageTotal = totalMarriages.filter($"level_1"==="Total")
marriageTotal.registerTempTable("marriageTotal")
display(sqlContext.sql("select * from marriageTotal where level_2 =='Same Age' or level_2 =='Groom Older By 2 Years' or level_2 == 'Groom Younger By 2 Years' or level_2 == 'Groom Older By 10 Years' or level_2 =='Groom Younger By 1 Year'" ))

// COMMAND ----------

// MAGIC %md ###### Based on the scatter plot I think a few concluding observations can be made.
// MAGIC * Marriages where the Groom was younger than the bride were less common compared to Same Age marriages
// MAGIC * Marriages where Groom was Older By 10 Years was the least common across the 5 categories we selected to review
// MAGIC * Marriages where the Groom was Older used to be higher in the 1980s & 1990s but from 2005 onwards, same Age marriages started to have a higher trend.
// MAGIC * The line graph seems to point towards a uptrend. A quick Google actually prove me right that number of marriages have been increasing too (https://www.straitstimes.com/singapore/more-marriages-took-place-in-2017-of-which-nearly-a-quarter-were-inter-ethnic)
// MAGIC 
// MAGIC ######Which  makes me wonder, if divorces seem to be increasing, why would marriage rates increase as well? Could it be because of the increase in marriages? Perhaps people realised that their first marriage wasn't the right one and they moved on the remarry someone else in the future. My gut tells me that number of remarriages would likely be higher too. So lets take a look at it using the other dataset available.

// COMMAND ----------

totalMarriagesOrder.registerTempTable("totalMarriagesOrder")
display(sqlContext.sql("select * from totalMarriagesOrder where level_1 =='Remarriages' or level_1=='First Marriages'" ))

// COMMAND ----------

// MAGIC %md ###### As seen we can see that remarriages have been experiencing a steady uptrend. First Marriages on the other hand seem to fluctuate quite a bit with time. But that doesn't necessarily mean that there has been a decline in First Marriages. Its interesting to note that in 2018, there was actually a decline in both First Marriages & Remarriages. Maybe the timing of marriage wasn't the most appropriate for couples in the year 2018 itself. Who knows. One thing for sure is that I believe after this, I have a better hang of the basic syntax and usage of the Scala language. So lets end this project here and hopefully move on to more projects where I'll get to use Scala's deep learning libraries eventually. Thanks for your time & cheers to helping the world using data science. :)

// COMMAND ----------

// MAGIC %md
// MAGIC END OF NOTEBOOK
// MAGIC -------------
