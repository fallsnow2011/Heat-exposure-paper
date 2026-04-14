// ====================================================
// Download Microsoft Global ML Building Footprints / 涓嬭浇 Microsoft Global ML Building Footprints
// Five study cities: London, Birmingham, Bristol, Manchester, Newcastle / 5 涓爺绌跺煄甯傦細London銆丅irmingham銆丅ristol銆丮anchester銆丯ewcastle
// The dataset includes building-height attributes / 鏁版嵁鍖呭惈寤虹瓚楂樺害灞炴€?// ====================================================

// Microsoft building dataset from the GEE Community Catalog / Microsoft 寤虹瓚鏁版嵁闆嗭紙GEE Community Catalog锛?var msBuildings = ee.FeatureCollection('projects/sat-io/open-datasets/MSBuildings/United_Kingdom');

// ==================== City boundary definition / 鍩庡競杈圭晫瀹氫箟 ====================
// Bounds from city_boundaries/*_boundary.geojson (EPSG:4326)
var london = ee.Geometry.Rectangle([-0.51028, 51.286759, 0.334024, 51.691876]);
var birmingham = ee.Geometry.Rectangle([-2.033641, 52.381083, -1.728853, 52.608669]);
var bristol = ee.Geometry.Rectangle([-2.718317, 51.397282, -2.510462, 51.544431]);
var manchester = ee.Geometry.Rectangle([-2.319897, 53.340122, -2.146848, 53.544604]);
var newcastle = ee.Geometry.Rectangle([-1.775668, 54.959995, -1.530845, 55.079389]);

// ==================== Processing function / 澶勭悊鍑芥暟 ====================
function processCity(cityGeom, cityName) {
  // Clip building footprints to the city extent / 鎸夊煄甯傝竟鐣岃鍓缓绛?  var buildings = msBuildings.filterBounds(cityGeom);

  // Print a quick building count / 缁熻
  var count = buildings.size();
  print(cityName + ' building count / 寤虹瓚鏁伴噺:', count);

  // Add the footprints to the map for inspection / 鍙鍖?  Map.addLayer(buildings.style({color: 'red', fillColor: '00000000', width: 1}),
               {}, cityName + ' Buildings');

  // Export as GeoJSON / 瀵煎嚭 GeoJSON
  Export.table.toDrive({
    collection: buildings,
    description: 'MS_Buildings_' + cityName,
    folder: 'GEE_Buildings',
    fileFormat: 'GeoJSON'
  });

  // Also export as Shapefile / 涔熷彲浠ュ鍑?Shapefile
  Export.table.toDrive({
    collection: buildings,
    description: 'MS_Buildings_SHP_' + cityName,
    folder: 'GEE_Buildings',
    fileFormat: 'SHP'
  });

  return buildings;
}

// ==================== Process all cities / 澶勭悊鎵€鏈夊煄甯?====================
var cities = [
  {geom: london, name: 'London'},
  {geom: birmingham, name: 'Birmingham'},
  {geom: bristol, name: 'Bristol'},
  {geom: manchester, name: 'Manchester'},
  {geom: newcastle, name: 'Newcastle'}
];

// Process each city one by one / 閫愪釜澶勭悊
for (var i = 0; i < cities.length; i++) {
  processCity(cities[i].geom, cities[i].name);
}

// Center the map view / 灞呬腑鏄剧ず
Map.setCenter(-1.5, 53, 6);

// ==================== Inspect dataset fields / 鏌ョ湅鏁版嵁缁撴瀯 ====================
// Inspect the attributes of one example building / 鏌ョ湅涓€涓ず渚嬪缓绛戠殑灞炴€?var sampleBuilding = msBuildings.filterBounds(london).first();
print('Sample building attributes / 绀轰緥寤虹瓚灞炴€?', sampleBuilding);

// ==================== Export notes / 璇存槑 ====================
print('========================================');
print('Export instructions / 瀵煎嚭璇存槑:');
print('1. Open the Tasks panel in the top-right corner / 鐐瑰嚮鍙充笂瑙?Tasks 闈㈡澘');
print('2. Click RUN for each city export task / 瀵规瘡涓煄甯傜偣鍑?RUN 鍚姩瀵煎嚭');
print('3. GeoJSON files will be saved to the GEE_Buildings folder in Google Drive / GeoJSON 鏂囦欢灏嗕繚瀛樺埌 Google Drive 鐨?GEE_Buildings 鏂囦欢澶?);
print('');
print('Key attributes / 鏁版嵁灞炴€?');
print('- geometry: building footprint polygon / 寤虹瓚杞粨澶氳竟褰?);
print('- height: building height in metres, estimated by Microsoft ML / 寤虹瓚楂樺害锛堢背锛夛紝鐢?Microsoft ML 浼扮畻');
print('- confidence: confidence score / 缃俊搴?);
print('========================================');



