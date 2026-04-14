
// ==================== City boundary definition / 鍩庡競杈圭晫瀹氫箟 ====================
var london = ee.Geometry.Rectangle([-0.51028, 51.286759, 0.334024, 51.691876]);
var birmingham = ee.Geometry.Rectangle([-2.033641, 52.381083, -1.728853, 52.608669]);
var bristol = ee.Geometry.Rectangle([-2.718317, 51.397282, -2.510462, 51.544431]);
var manchester = ee.Geometry.Rectangle([-2.319897, 53.340122, -2.146848, 53.544604]);
var newcastle = ee.Geometry.Rectangle([-1.775668, 54.959995, -1.530845, 55.079389]);

// ==================== Function for a single city / 澶勭悊鍗曚釜鍩庡競鐨勫嚱鏁?====================
function processERA5City(cityGeom, cityName) {
  // ERA5-Land Daily Aggregated
  var era = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
    .select('temperature_2m_max')  // daily maximum 2 m air temperature / 鏃ユ渶楂?2 m 姘旀俯
    .filterDate('2022-06-01', '2022-08-31')
    .filterBounds(cityGeom);

  // Compute the mean Tmax within the city extent for each day / 璁＄畻姣忎竴澶╁煄甯傝寖鍥村唴鐨勫钩鍧?Tmax
  var daily = era.map(function(img) {
    var tMean = img.reduceRegion({
      reducer: ee.Reducer.mean(),
      geometry: cityGeom,
      scale: 11132,   // ERA5-Land resolution is about 11 km / ERA5-Land 鍒嗚鲸鐜囩害 11 km
      maxPixels: 1e9
    }).get('temperature_2m_max');

    // Convert to Celsius and preserve the date attribute / 杞负 鈩?骞朵繚鐣欐棩鏈熷睘鎬?
    return ee.Feature(null, {
      'date': img.date().format('YYYY-MM-dd'),
      'Tmax_K': tMean,
      'Tmax_C': ee.Number(tMean).subtract(273.15),
      'city': cityName
    });
  });

  return ee.FeatureCollection(daily);
}

// ==================== Process all cities / 澶勭悊鎵€鏈夊煄甯?====================
var cities = [
  {geom: london, name: 'London'},
  {geom: birmingham, name: 'Birmingham'},
  {geom: bristol, name: 'Bristol'},
  {geom: manchester, name: 'Manchester'},
  {geom: newcastle, name: 'Newcastle'}
];

// Merge all city results into one table for downstream use / 灏嗘墍鏈夊煄甯傜粨鏋滃悎骞朵负涓€寮犺〃锛屼究浜庡悗缁鐞?
var allCities = ee.FeatureCollection([]);

for (var i = 0; i < cities.length; i++) {
  var cityData = processERA5City(cities[i].geom, cities[i].name);

  // Print a quick preview / 鎵撳嵃棰勮
  print(cities[i].name + ' - ERA5 Daily Tmax (first 5 days / 鍓?5 澶?:', cityData.limit(5));

  // Export a dedicated CSV for each city / 鍗曠嫭瀵煎嚭姣忎釜鍩庡競鐨?CSV
  Export.table.toDrive({
    collection: cityData,
    description: 'ERA5Land_daily_Tmax_2022_' + cities[i].name,
    folder: 'GEE_ERA5_Tmax',
    fileFormat: 'CSV'
  });

  // Merge into the combined table / 鍚堝苟鍒版€昏〃
  allCities = allCities.merge(cityData);
}

// Export the combined CSV for all cities (optional fallback) / 瀵煎嚭鎵€鏈夊煄甯傚悎骞剁殑 CSV锛堝閫夛級
Export.table.toDrive({
  collection: allCities,
  description: 'ERA5Land_daily_Tmax_2022_AllCities',
  folder: 'GEE_ERA5_Tmax',
  fileFormat: 'CSV'
});

// ==================== Visualize one example day / 鍙鍖栨煇涓€澶╃殑娓╁害 ====================
// Example: 2022-08-11, near the London heatwave peak / 浠?2022-08-11 涓轰緥锛堜鸡鏁︾儹娴珮宄帮級
var heatwaveDay = ee.Image(ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
    .select('temperature_2m_max')
    .filterDate('2022-08-11', '2022-08-12')
    .first())
    .subtract(273.15);  // convert to Celsius / 杞负 鈩?
Map.addLayer(heatwaveDay.clip(london),
    {min: 30, max: 45, palette: ['blue', 'yellow', 'red']},
    'ERA5 Tmax 2022-08-11 London');

Map.setCenter(-0.1, 51.5, 9);

print('========================================');
print('Export instructions / 瀵煎嚭璇存槑:');
print('1. Open the Tasks panel in the top-right corner / 鐐瑰嚮鍙充笂瑙?Tasks 闈㈡澘');
print('2. Click RUN for each city CSV export / 瀵规瘡涓煄甯傜殑 CSV 鐐瑰嚮 RUN 鍚姩瀵煎嚭');
print('3. Files will be saved to the GEE_ERA5_Tmax folder in Google Drive / 鏂囦欢灏嗕繚瀛樺埌 Google Drive 鐨?GEE_ERA5_Tmax 鏂囦欢澶?);
print('========================================');
print('');
print('Known heatwave dates for reference / 宸茬煡鐑氮鏃ユ湡锛堜緵鍙傝€冿級:');
print('London: 2022-08-11 ~ 2022-08-14');
print('Birmingham: 2022-07-17~19, 2022-08-10~14');
print('Bristol: 2022-07-17~19, 2022-08-10~14');
print('Manchester: 2022-07-17~19, 2022-08-10~14');
print('Newcastle: 2022-07-17~19, 2022-08-10~14');



