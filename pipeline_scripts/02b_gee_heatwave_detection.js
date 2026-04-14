
// ================= 1. Configuration / 閰嶇疆鍖哄煙 =================

var level1 = ee.FeatureCollection("FAO/GAUL/2015/level1"); // First-level administrative units (for London) / 涓€绾ц鏀垮尯锛堢敤浜?London锛?var level2 = ee.FeatureCollection("FAO/GAUL/2015/level2"); // Second-level administrative units (for other cities) / 浜岀骇琛屾斂鍖猴紙鐢ㄤ簬鍏朵粬鍩庡競锛?
var startDate = '2022-01-01';
var endDate = '2022-12-31';

// UK Met Office heatwave thresholds / UK Met Office 鐑氮闃堝€煎畾涔?// Source / 鏉ユ簮: https://www.metoffice.gov.uk/weather/learn-about/weather/types-of-weather/temperature/heatwave
var cityConfigs = [
  {
    name: 'London',
    threshold: 28,  // Met Office heatwave threshold for London / Met Office锛歀ondon 鐑氮闃堝€?    collection: 'level1',
    coords: [-0.1276, 51.5072]
  },
  {
    name: 'Birmingham',
    threshold: 26,  // Met Office heatwave threshold for the Midlands / Met Office锛歁idlands 鐑氮闃堝€?    collection: 'level2',
    coords: [-1.8904, 52.4862]
  },
  {
    name: 'Bristol',
    threshold: 26,  // Met Office heatwave threshold for the Southwest / Met Office锛歋outhwest 鐑氮闃堝€?    collection: 'level2',
    coords: [-2.5879, 51.4545]
  },
  {
    name: 'Manchester',
    threshold: 25,  // Met Office heatwave threshold for the Northwest / Met Office锛歂orthwest 鐑氮闃堝€?    collection: 'level2',
    coords: [-2.2426, 53.4808]
  },
  {
    name: 'Newcastle',
    threshold: 25,  // Met Office heatwave threshold for the Northeast / Met Office锛歂ortheast 鐑氮闃堝€?    collection: 'level2',
    coords: [-1.6178, 54.9783]
  }
];

// ================= 2. Core extraction function / 鏍稿績鏁版嵁鎻愬彇鍑芥暟 =================

var getCityDataRequest = function(config) {
  var collection = (config.collection === 'level1') ? level1 : level2;
  var point = ee.Geometry.Point(config.coords);
  var feature = collection.filterBounds(point).first();

  var aoi = ee.Algorithms.If(
    feature,
    feature.geometry().simplify(1000),
    point.buffer(5000)
  );

  var era = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
    .select('temperature_2m_max')
    .filterDate(startDate, endDate)
    .filterBounds(aoi);

  var dailySeries = era.map(function(img) {
    var dateStr = img.date().format('YYYY-MM-dd');
    var tempK = img.reduceRegion({
      reducer: ee.Reducer.mean(),
      geometry: aoi,
      scale: 25000,
      maxPixels: 1e9,
      bestEffort: true
    }).get('temperature_2m_max');

    var tempC = ee.Algorithms.If(
      ee.Algorithms.IsEqual(tempK, null),
      null,
      ee.Number(tempK).subtract(273.15)
    );

    return ee.Feature(null, {'tmax': tempC, 'date': dateStr});
  }).filter(ee.Filter.notNull(['tmax']));

  var p95 = ee.Algorithms.If(
    dailySeries.size().gt(0),
    ee.Number(dailySeries.reduceColumns(ee.Reducer.percentile([95]), ['tmax']).get('p95')),
    0
  );

  var maxFeat = dailySeries.sort('tmax', false).first();

  // Heatwave filter: Tmax >= regional threshold AND Tmax >= P95 / 鐑氮绛涢€夋潯浠讹細Tmax >= 鍖哄煙闃堝€?AND Tmax >= P95
  var threshold = config.threshold;
  var hotDates = dailySeries.filter(ee.Filter.and(
      ee.Filter.gte('tmax', threshold),
      ee.Filter.gte('tmax', p95)
  )).aggregate_array('date');

  return ee.Dictionary({
    name: config.name,
    threshold: threshold,
    p95: p95,
    max_val: ee.Algorithms.If(maxFeat, maxFeat.get('tmax'), 0),
    max_date: ee.Algorithms.If(maxFeat, maxFeat.get('date'), 'No Data'),
    hot_dates: hotDates,
    data_valid: dailySeries.size().gt(0)
  });
};

// ================= 3. Consecutive-day check (>= 3 days) / 杩炵画鎬ф鏌ワ紙>= 3 澶╋級 =================

var checkConsecutive = function(dateList) {
  if (!dateList || dateList.length < 3) return [];
  var parseTime = function(d) { return new Date(d).getTime(); };
  var oneDay = 86400000;

  var events = [];
  var currentSeq = [dateList[0]];

  for (var i = 1; i < dateList.length; i++) {
    var t1 = parseTime(dateList[i-1]);
    var t2 = parseTime(dateList[i]);
    if (Math.abs(t2 - t1 - oneDay) < (oneDay * 0.5)) {
      currentSeq.push(dateList[i]);
    } else {
      if (currentSeq.length >= 3) events.push(currentSeq.join(', '));
      currentSeq = [dateList[i]];
    }
  }
  if (currentSeq.length >= 3) events.push(currentSeq.join(', '));
  return events;
};

// ================= 4. Serial processing / 涓茶澶勭悊 =================

var processNextCity = function(index) {
  if (index >= cityConfigs.length) {
    print('鉁?Finished all city calculations / 鎵€鏈夊煄甯傝绠楀畬鎴愶紒');
    return;
  }

  var config = cityConfigs[index];
  print('鈴?[' + (index + 1) + '/' + cityConfigs.length + '] Requesting data for ' + config.name + ' / 姝ｅ湪璇锋眰 ' + config.name + ' 鐨勬暟鎹?..');

  var request = getCityDataRequest(config);

  request.evaluate(function(result, error) {
    if (error) {
      print('鉂?Error while processing ' + config.name + ' / ' + config.name + ' 璁＄畻鎶ラ敊: ' + error);
    } else if (!result || !result.data_valid) {
      print('鉂?No data returned for ' + config.name + ' / ' + config.name + ' 鏃犳暟鎹繑鍥?);
    } else {
      var name = result.name;
      var thr = result.threshold;
      var p95 = Number(result.p95).toFixed(2);
      var maxVal = Number(result.max_val).toFixed(2);
      var maxDate = result.max_date;
      var hotDates = result.hot_dates;
      var events = checkConsecutive(hotDates);

      print('----------------------------------------------');
      print('馃彊锔? ' + name.toUpperCase() + ' (Met Office threshold / Met Office 闃堝€? ' + thr + '掳C)');
      print('馃搳  P95 threshold / P95 闃堝€? ' + p95 + '掳C');

      if (events.length > 0) {
        print('馃敟  Heatwave detected / 绗﹀悎鐑氮瀹氫箟锛堣繛缁?>= 3 澶╋級');
        events.forEach(function(e) { print('    鉃★笍 ' + e); });
      } else {
        print('鉂勶笍  No heatwave detected / 鏈娴嬪埌鐑氮');
        print('    馃専 Annual maximum temperature / 鍏ㄥ勾鏈€楂樻俯: ' + maxDate + ' (' + maxVal + '掳C)');
      }
    }
    processNextCity(index + 1);
  });
};

// ================= 5. Run script / 杩愯 =================
print('馃殌 Heatwave detection with UK Met Office regional thresholds / 鐑氮鏃ョ瓫閫夛細UK Met Office 鍖哄煙闃堝€兼爣鍑?);
print('Data source / 鏁版嵁婧? ERA5-Land Daily Aggregated (temperature_2m_max)');
print('Heatwave definition / 鐑氮瀹氫箟: at least 3 consecutive days, Tmax 鈮?regional threshold AND Tmax 鈮?P95 / 杩炵画 鈮?3 澶╋紝Tmax 鈮?鍖哄煙闃堝€?涓?Tmax 鈮?P95');
print('');
processNextCity(0);

/*
================= 2022 heatwave screening results / 2022 骞寸儹娴棩绛涢€夌粨鏋?=================

| City / 鍩庡競 | Met Office threshold / Met Office 闃堝€?| P95 | Heatwave events / 鐑氮浜嬩欢 |
|------------|----------------|--------|---------------------------------------------|
| London     | 28掳C           | 23.18掳C| 2022-08-11 ~ 08-14 (4 days / 4 澶?         |
| Birmingham | 26掳C           | 24.07掳C| 07-17~19 (3 days / 3 澶? + 08-10~14 (5 days / 5 澶? |
| Bristol    | 26掳C           | 24.37掳C| 07-17~19 (3 days / 3 澶? + 08-08~14 (7 days / 7 澶? |
| Manchester | 25掳C           | 21.35掳C| 07-17~19 (3 days / 3 澶? + 08-10~14 (5 days / 5 澶? |
| Newcastle  | 25掳C           | 22.42掳C| 07-17~19 (3 days / 3 澶? + 08-10~12 (3 days / 3 澶? |

References / 鍙傝€冩枃鐚?
- UK Met Office Heatwave Definition:
  https://www.metoffice.gov.uk/weather/learn-about/weather/types-of-weather/temperature/heatwave
- ERA5-Land: Mu帽oz Sabater, J., (2019). ERA5-Land hourly data from 1981 to present.
  Copernicus Climate Change Service (C3S) Climate Data Store (CDS).
*/



