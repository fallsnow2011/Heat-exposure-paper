
// ==================== City boundary definition / 鍩庡競杈圭晫瀹氫箟 ====================
// Method 1: use rectangular extents (simple and fast) / 鏂规硶 1锛氫娇鐢ㄧ煩褰㈣寖鍥达紙绠€鍗曞揩閫燂級
var london = ee.Geometry.Rectangle([-0.51028, 51.286759, 0.334024, 51.691876]);
var birmingham = ee.Geometry.Rectangle([-2.033641, 52.381083, -1.728853, 52.608669]);
var bristol = ee.Geometry.Rectangle([-2.718317, 51.397282, -2.510462, 51.544431]);
var manchester = ee.Geometry.Rectangle([-2.319897, 53.340122, -2.146848, 53.544604]);
var newcastle = ee.Geometry.Rectangle([-1.775668, 54.959995, -1.530845, 55.079389]);

// Method 2: if you uploaded city-boundary shapefiles, you can use them here / 鏂规硶 2锛氬鏋滃凡涓婁紶鍩庡競杈圭晫 shapefile锛屽彲鍦ㄦ浣跨敤
// var london = ee.FeatureCollection('users/your_name/London_boundary').geometry();

// ==================== Cloud-masking function / 浜戞帺鑶滃嚱鏁?====================
function maskLandsatC2(image) {
  var qa = image.select('QA_PIXEL');
  var cloudBitMask  = (1 << 3);  // Bit 3: cloud
  var shadowBitMask = (1 << 4);  // Bit 4: cloud shadow
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
               .and(qa.bitwiseAnd(shadowBitMask).eq(0));
  return image.updateMask(mask);
}

// ==================== LST conversion function / LST 杞崲鍑芥暟 ====================
// Convert ST_B10 to degrees Celsius / 灏?ST_B10 杞负鎽勬皬搴?
// Formula: ST(K) = DN * 0.00341802 + 149.0, then subtract 273.15 to convert to Celsius / 鍏紡锛歋T(K) = DN * 0.00341802 + 149.0锛岀劧鍚庡噺 273.15 杞负 鈩?
function addLST(image) {
  var lstK = image.select('ST_B10')
    .multiply(0.00341802).add(149.0);  // Kelvin
  var lstC = lstK.subtract(273.15).rename('LST_C');
  return image.addBands(lstC);
}

// ==================== Function for a single city / 澶勭悊鍗曚釜鍩庡競鐨勫嚱鏁?====================
function processCity(cityGeom, cityName) {
  // Merge Landsat 8 and 9 Level-2 collections / 鍚堝苟 Landsat 8 鍜?9 鐨?Level-2 鏁版嵁
  var l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2');
  var l9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2');

  var ls = l8.merge(l9)
    .filterBounds(cityGeom)
    .filterDate('2018-06-01', '2023-08-31')
    .filter(ee.Filter.calendarRange(6, 8, 'month'))  // June-August only / 浠呬繚鐣?6-8 鏈?
    .filter(ee.Filter.lt('CLOUD_COVER', 30));  // prefilter scenes with cloud cover < 30% / 浜戦噺 < 30% 棰勭瓫閫?

  // Apply cloud masking and convert to temperature / 搴旂敤浜戞帺鑶滃苟杞崲娓╁害
  var ls_processed = ls
    .map(maskLandsatC2)
    .map(addLST);

  // Compute the summer median LST (30 m) / 璁＄畻澶忓涓綅鏁?LST锛?0 m锛?
  var lst_median = ls_processed.select('LST_C').median().clip(cityGeom);

  // Summary statistics / 缁熻淇℃伅
  var stats = lst_median.reduceRegion({
    reducer: ee.Reducer.mean().combine({
      reducer2: ee.Reducer.stdDev(),
      sharedInputs: true
    }).combine({
      reducer2: ee.Reducer.minMax(),
      sharedInputs: true
    }),
    geometry: cityGeom,
    scale: 30,
    maxPixels: 1e13
  });

  print(cityName + ' - Available scenes / 鍙敤褰卞儚鏁伴噺:', ls.size());
  print(cityName + ' - LST statistics / LST 缁熻:', stats);

  return {
    image: lst_median,
    name: cityName,
    geometry: cityGeom
  };
}

// ==================== Process all cities / 澶勭悊鎵€鏈夊煄甯?====================
var cities = [
  {geom: london, name: 'London'},
  {geom: birmingham, name: 'Birmingham'},
  {geom: bristol, name: 'Bristol'},
  {geom: manchester, name: 'Manchester'},
  {geom: newcastle, name: 'Newcastle'}
];

// Process and visualize all cities / 澶勭悊骞跺彲瑙嗗寲鎵€鏈夊煄甯?var palette = ['blue', 'cyan', 'yellow', 'orange', 'red'];

for (var i = 0; i < cities.length; i++) {
  var result = processCity(cities[i].geom, cities[i].name);

  // Add the raster layer to the map / 娣诲姞鍒板湴鍥?  Map.addLayer(result.image,
    {min: 25, max: 45, palette: palette},
    'LST_' + result.name);

  // Export the city baseline as GeoTIFF / 瀵煎嚭 GeoTIFF
  Export.image.toDrive({
    image: result.image.toFloat(),
    description: 'LST_median_summer_2018_2023_' + result.name + '_30m',
    folder: 'GEE_LST_Baseline',
    region: result.geometry,
    scale: 30,
    crs: 'EPSG:27700',  // British National Grid
    maxPixels: 1e13
  });
}

// Center the map view on the UK / 灞呬腑鍒拌嫳鍥?Map.setCenter(-1.5, 52.5, 6);

print('========================================');
print('Export instructions / 瀵煎嚭璇存槑:');
print('1. Open the Tasks panel in the top-right corner / 鐐瑰嚮鍙充笂瑙?Tasks 闈㈡澘');
print('2. Click RUN for each city export task / 瀵规瘡涓煄甯傜偣鍑?RUN 鍚姩瀵煎嚭');
print('3. Files will be saved to the GEE_LST_Baseline folder in Google Drive / 鏂囦欢灏嗕繚瀛樺埌 Google Drive 鐨?GEE_LST_Baseline 鏂囦欢澶?);
print('========================================');



