import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.{Bucketizer, StringIndexer}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.collection.mutable

case class DTPreprocessFmt(
                         localMode: Boolean = false,
                         var inputDir: String = "",
                         comboPluPath: String = "tlogComboPluAll0122",
                         orderPluPath: String = "tlogPluAll0231",
                         popularPath: String = "top_pluall_0231",
                         var weatherPath: String = "",
                         srPath: String = "sr_replace.csv",
                         storeWeatherPath: String = "bkid_weather.csv",
                         drinkGrouppath: String = "drink_group.csv",
                         outputPath: String = "tlogFmt0122"
                       )

object DTPreprocessSpark {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN)

    // Construct spark session
    val params = DTPreprocessFmt()
    val spark = params.localMode match {
      case true => {
        params.inputDir = "./data/"
        params.weatherPath = "./data/weather/*HOURLY*"
        SparkSession.builder()
          .appName("DTPreprocess")
          .master("local")
          .config("spark.driver.memory", "30g")
          .config("spark.driver.maxResultSize", "15g")
          .getOrCreate()
      }
      case false => {
        params.inputDir = "s3://path/data/"
        params.weatherPath = "s3://path/data/weather/*HOURLY*.csv"
        SparkSession.builder()
          .appName("DTPreprocess")
          .config("spark.driver.maxResultSize", "15g")
          .getOrCreate()
      }
    }
    val ts = to_timestamp(trim(col("DATA DATE and HOUR")), "dd/MMM/yyyy HH")

    val bkidWeather = spark.read.option("header", "true").csv(params.weatherPath).distinct()
    val bkidWeather1 = bkidWeather.select("DATA DATE and HOUR", "WEATHER STATION CODE",
      bkidWeather.columns(7), "WBI WEATHER TEXT")
      .withColumnRenamed("WEATHER STATION CODE", "STATION")
      .withColumnRenamed(bkidWeather.columns(7), "feels")
      .withColumn("hrStamp", ts)
      .withColumn("hrStamp", date_format(col("hrStamp"), "dd/MMM/yyyy HH"))

    bkidWeather1.printSchema()
    bkidWeather1.show(50, false)

    val storeLookup = spark.read.option("header", "true")
      .csv(params.inputDir + params.storeWeatherPath).distinct()
    val storeWeather = bkidWeather1.join(storeLookup, Seq("STATION"), "inner")
      .withColumnRenamed("BK#", "bkid")
      .withColumn("feels", col("feels").cast(IntegerType))
      .withColumnRenamed("WBI WEATHER TEXT", "weather")
      .distinct()

    storeWeather.printSchema()
    storeWeather.show(20, false)

    val popular = spark.read.option("header", "true").csv(params.inputDir + params.popularPath)
      .withColumnRenamed("plunum", "plu")
      .filter(col("count") >= 2920)
      .select("plu").distinct()
    popular.show()

    val orderPlu = spark.read.option("timestampFormat", "yyyy-MM-dd hh:mm:ss").json(params.inputDir + params.orderPluPath)
      .withColumn("plu", trim(col("plunum")))
      .withColumnRenamed("tlog_sale_header_uid", "uuid")
      .withColumnRenamed("rest_no", "bkid")
      .withColumnRenamed("unit_price", "price")
      .join(popular, Seq("plu"), "inner")
      .withColumn("hrStamp", date_format(col("transaction_start_datetime"), "dd/MMM/yyyy HH"))
      .join(storeWeather, Seq("bkid", "hrStamp"), "inner")
      .withColumn("time", date_format(col("transaction_start_datetime"), "EEE-HH"))
      .select("uuid","plu", "price", "bkid", "time", "feels", "weather")

    orderPlu.orderBy("time", "uuid").show(200, false)

    val splits = Array.range(0, 100, 5).map(_.toDouble)
    splits.foreach(
      println(_)
    )
    val splits1 = Double.NegativeInfinity +: splits :+ Double.PositiveInfinity
    val bucketizer = new Bucketizer().setInputCol("feels").setOutputCol("feelsBucket").setSplits(splits1)

    val orderPlu1 = bucketizer.transform(orderPlu).drop("feels")

    orderPlu1.show(20, false)
    orderPlu1.printSchema()

    val plusIndexer = new StringIndexer().setHandleInvalid("skip").setInputCol("plu").setOutputCol("pluidx")
    val bkidIndexer = new StringIndexer().setHandleInvalid("skip").setInputCol("bkid").setOutputCol("bkidx")
    val timeIndexer = new StringIndexer().setHandleInvalid("skip").setInputCol("time").setOutputCol("timeidx")
    val weatherIndexer = new StringIndexer().setHandleInvalid("skip").setInputCol("weather").setOutputCol("weatheridx")
    val plusIndexerModel = plusIndexer.fit(orderPlu1)
    val bkidIndexerModel = bkidIndexer.fit(orderPlu1)
    val timeIndexerModel = timeIndexer.fit(orderPlu1)
    val weatherIndexerModel = weatherIndexer.fit(orderPlu1)


    val orderPluIndexed = plusIndexerModel.transform(orderPlu1).withColumn("pluidx", col("pluidx") + 1)
    orderPluIndexed.select("plu", "pluidx").distinct().coalesce(1).write.mode("overwrite").csv(params.inputDir + "pluLookup")
    val orderBkIndexed = bkidIndexerModel.transform(orderPluIndexed)
    orderBkIndexed.select("bkid", "bkidx").distinct().coalesce(1).write.mode("overwrite").csv(params.inputDir + "bkLookup")
    val orderTimeIndexed = timeIndexerModel.transform(orderBkIndexed)
    orderTimeIndexed.select("time", "timeidx").distinct().coalesce(1).write.mode("overwrite").csv(params.inputDir + "timeLookup")
    val orderWeatherIndexed = weatherIndexerModel.transform(orderTimeIndexed)
    orderWeatherIndexed.select("weather", "weatheridx").distinct().coalesce(1).write.mode("overwrite").csv(params.inputDir + "weatherLookup")
    val orderIndexed = orderWeatherIndexed.drop("plu", "bkid", "time", "weather")

    orderIndexed.show()
    orderIndexed.printSchema()

    def prePadding: mutable.WrappedArray[java.lang.Double] => Array[Float] = x => {
      if (x.array.size < 5) x.array.map(_.toFloat).reverse.padTo(5, 0f).reverse
      else x.array.map(_.toFloat).takeRight(5)
    }
    val prePaddingUDF = udf(prePadding)

    def slideSession(df: DataFrame, sessionLength: Int): DataFrame = {
      val sqlContext = df.sqlContext
      import sqlContext.implicits._
      val r = new scala.util.Random
      val dfSlided = df.rdd.flatMap(x => {
        val session = x.getAs[mutable.WrappedArray[java.lang.Double]]("pluids")
          .array.map(_.toString).map(p => p + "-" + r.nextInt(100))
        val bkidx = x.getAs[Double]("bkidx")
        val timeidx = x.getAs[Double]("timeidx")
        val feelsBucket = x.getAs[Double]("feelsBucket")
        val weatheridx = x.getAs[Double]("weatheridx")
        val featureLabel = for (label <- session.slice(1, session.size)) yield {
          val endIdx = session.indexOf(label)
          val beginIdx = if (session.size <= sessionLength) 0 else endIdx - sessionLength
          val feature1 = session.slice(beginIdx, endIdx).map(pf => pf.split("-")(0).toFloat)
          val labelFmt = label.split("-")(0).toFloat
          (feature1, labelFmt, bkidx, timeidx, feelsBucket, weatheridx)
        }
        featureLabel
      }).toDF("pluids", "label", "bkidx", "timeidx", "feelsBucket", "weatheridx").na.drop()
      dfSlided
    }

    val orderIndexed1 = orderIndexed.orderBy(col("uuid"), col("price").desc)
      .groupBy("uuid", "bkidx", "timeidx", "feelsBucket", "weatheridx").agg(collect_list(col("pluidx")).as("pluids"))
      .select("bkidx", "timeidx", "feelsBucket", "weatheridx", "pluids")
      .filter(size(col("pluids")) > 1)

    orderIndexed1.count()
    orderIndexed1.cache()
    orderIndexed1.show(20, false)

    val orderIndexed2 = slideSession(orderIndexed1, 5).withColumn("pluids", prePaddingUDF(col("pluids")))

    orderIndexed2.show(20, false)

    orderIndexed2.write.mode("overwrite").json(params.inputDir + params.outputPath)

    spark.close()
  }
}
