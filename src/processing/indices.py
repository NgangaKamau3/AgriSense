import ee

def calculate_ndvi(image):
    """Calculates Normalized Difference Vegetation Index (NDVI)."""
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return image.addBands(ndvi)

def calculate_ndwi(image):
    """Calculates Normalized Difference Water Index (NDWI)."""
    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
    return image.addBands(ndwi)

def calculate_ndmi(image):
    """Calculates Normalized Difference Moisture Index (NDMI)."""
    ndmi = image.normalizedDifference(['B8', 'B11']).rename('NDMI')
    return image.addBands(ndmi)

def calculate_msi(image):
    """Calculates Moisture Stress Index (MSI)."""
    # MSI = B11 / B8
    msi = image.select('B11').divide(image.select('B8')).rename('MSI')
    return image.addBands(msi)

def calculate_ndre(image):
    """Calculates Normalized Difference Red Edge (NDRE)."""
    # NDRE = (B8 - B5) / (B8 + B5)
    ndre = image.normalizedDifference(['B8', 'B5']).rename('NDRE')
    return image.addBands(ndre)

def add_all_indices(image):
    """Adds all supported indices to the image."""
    image = calculate_ndvi(image)
    image = calculate_ndwi(image)
    image = calculate_ndmi(image)
    image = calculate_msi(image)
    image = calculate_ndre(image)
    return image
