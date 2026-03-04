[[Image processing]]
# Camera Technical Specifications Guide for Defect Detection
## Complete Reference for Industrial Vision Camera Selection

---

## PAGE 1: FUNDAMENTAL CAMERA SPECIFICATIONS

### 1. SENSOR RESOLUTION (Megapixels)

**Definition:** The total number of pixels the camera sensor can capture, typically expressed in megapixels (MP). Each pixel is a light-sensitive element that records image information.

**How It's Specified:**
- Horizontal × Vertical pixels (e.g., 2448 × 2048 = 5.0 MP)
- Total megapixels (e.g., 5 MP, 12 MP, 20 MP)

**Why It Matters for Defect Detection:**
Resolution determines the smallest defect you can reliably detect. Higher resolution allows you to see smaller defects or inspect larger areas with the same detail.

**Calculation Method:**

```
Required Resolution = (Field of View / Smallest Defect Size) × Pixels Per Defect

Where:
- Field of View (FOV) = Area you need to inspect (mm × mm)
- Smallest Defect Size = Minimum defect you must detect (mm)
- Pixels Per Defect = Recommended 5-10 pixels minimum for reliable detection
```

**Practical Example:**

```
Inspection Requirement:
- Product size: 100mm × 100mm
- Smallest defect: 0.5mm scratch
- Desired pixels per defect: 10 pixels (for clear detection)

Calculation:
- Pixels needed across width = (100mm / 0.5mm) × 10 = 2,000 pixels
- Pixels needed across height = (100mm / 0.5mm) × 10 = 2,000 pixels
- Total resolution needed = 2,000 × 2,000 = 4 MP

Recommendation: Select 5 MP camera (2448×2048) with safety margin
```

**Resolution Categories:**

| Resolution | Pixel Dimensions | Typical Use Cases |
|-----------|-----------------|-------------------|
| VGA | 640×480 (0.3 MP) | Simple presence/absence, low-detail inspection |
| 1 MP | 1280×1024 | Basic defect detection, barcode reading |
| 2 MP | 1920×1080 | Standard quality inspection |
| 5 MP | 2448×2048 | High-detail defect detection |
| 12 MP | 4096×3000 | Very small defects, large FOV |
| 20+ MP | 5120×4096+ | Ultra-fine inspection, semiconductor |

**Critical Considerations:**

- **More resolution ≠ Always better:** Higher resolution means:
  - Larger file sizes (more storage needed)
  - Slower processing (longer inference time)
  - More expensive cameras
  - More lighting needed

- **Balance resolution with throughput:** If you need to inspect 60 products/minute, a 20MP image may process too slowly

**What to Specify:**
- Minimum resolution required: _____ MP
- Horizontal × Vertical pixels preferred: _____ × _____
- Maximum resolution acceptable (processing limit): _____ MP

---

### 2. SENSOR SIZE (Physical Dimensions)

**Definition:** The physical size of the camera's image sensor, measured diagonally in inches (e.g., 1/2.3", 2/3", 1").

**Common Sensor Sizes:**
- 1/3" (4.8mm diagonal) - Small, inexpensive
- 1/2.5" (7.2mm diagonal) - Common in industrial cameras
- 1/2" (8.0mm diagonal) - Good balance
- 2/3" (11mm diagonal) - Professional industrial
- 1" (16mm diagonal) - High-end
- APS-C (23.6mm diagonal) - DSLR-grade
- Full Frame (43mm diagonal) - Maximum quality

**Why It Matters:**

1. **Light Gathering Ability:**
   - Larger sensor = larger individual pixels = more light collected
   - Better performance in low light
   - Higher signal-to-noise ratio

2. **Lens Compatibility:**
   - Sensor size determines required lens mount and optical format
   - Larger sensors need larger, more expensive lenses

3. **Pixel Size Calculation:**
   ```
   Pixel Size (μm) = Sensor Width (mm) / Horizontal Resolution (pixels)
   
   Example:
   1/2" sensor (6.4mm wide) with 2448 pixels horizontally
   Pixel size = 6.4mm / 2448 = 2.6 μm
   ```

**Relationship Between Sensor Size and Image Quality:**

| Sensor Size | Typical Pixel Size | Light Sensitivity | Cost |
|------------|-------------------|------------------|------|
| 1/3" | 1.6-2.2 μm | Low | $ |
| 1/2" | 2.2-3.0 μm | Medium | $$ |
| 2/3" | 3.0-4.5 μm | Good | $$$ |
| 1" | 4.5-6.0 μm | Excellent | $$$$ |

**What to Specify:**
- Required sensor size: _____ inches
- Minimum pixel size for light sensitivity: _____ μm
- Lens mount compatibility: C-mount / CS-mount / Other

---

### 3. FRAME RATE (Frames Per Second - FPS)

**Definition:** The number of complete images the camera can capture per second, measured in frames per second (fps) or Hertz (Hz).

**Common Frame Rates:**
- 10-30 fps: Stationary inspection, manual product placement
- 30-60 fps: Standard conveyor speeds
- 60-120 fps: Fast conveyors, high-speed inspection
- 120-500+ fps: Ultra-high-speed production lines, motion analysis

**Why It Matters:**

1. **Throughput Matching:**
   ```
   Required Frame Rate = Production Rate + Safety Margin
   
   Example:
   Production: 60 products/minute = 1 product/second
   Safety margin: 20%
   Required FPS: 1.2 fps minimum
   
   However: Practical minimum is 10-15 fps for reliable triggering
   ```

2. **Motion Blur Prevention:**
   - Higher frame rate allows shorter exposure time
   - Reduces blur on moving products
   - Critical for conveyor-based inspection

**Frame Rate Limitations:**

Frame rate depends on several factors:

```
Maximum Frame Rate = Interface Bandwidth / Image Data Size

Where Image Data Size = Resolution × Bit Depth × Color Channels

Example:
5MP (2448×2048), 8-bit, mono camera
Image size = 2448 × 2048 × 1 byte = 5 MB per frame

USB 3.0 bandwidth = 400 MB/s
Max FPS = 400 MB/s / 5 MB = 80 fps

GigE bandwidth = 125 MB/s  
Max FPS = 125 MB/s / 5 MB = 25 fps
```

**Frame Rate vs Exposure Time:**

For moving objects, you need both high frame rate AND short exposure:

```
Maximum Exposure Time = Acceptable Blur / Object Speed

Example:
Conveyor speed: 1 meter/second
Acceptable blur: 0.5mm
Max exposure = 0.5mm / 1000mm/s = 0.0005s = 0.5ms = 500μs

Therefore: Must use exposure ≤ 500μs
```

**Region of Interest (ROI) for Higher Frame Rates:**

If full resolution isn't needed, you can read only part of the sensor:

```
ROI Frame Rate = Full Frame Rate × (Full Height / ROI Height)

Example:
Full 5MP (2448×2048) at 30 fps
If using 2448×512 ROI (quarter height):
ROI FPS = 30 × (2048/512) = 120 fps
```

**What to Specify:**
- Minimum frame rate required: _____ fps
- Maximum frame rate needed: _____ fps
- Exposure time range: _____ μs to _____ ms
- Conveyor speed (if applicable): _____ m/s or products/min

---

### 4. SHUTTER TYPE: Global vs Rolling

**Definition:** The method by which the camera sensor captures light from all pixels.

**GLOBAL SHUTTER:**

**How It Works:**
- All pixels capture light simultaneously
- Electronic shutter opens/closes at exact same instant
- Entire frame is frozen at one moment in time

**Advantages:**
- No motion distortion
- Perfect for moving objects
- Accurate dimensional measurement
- Synchronized multi-camera setups

**Disadvantages:**
- More expensive
- Slightly lower light sensitivity
- More complex sensor design

**When Required:**
- Inspecting products on moving conveyors
- Any object motion during capture
- Rotating or vibrating products
- Multi-camera synchronized inspection
- High-speed inspection

**ROLLING SHUTTER:**

**How It Works:**
- Pixels are read line-by-line from top to bottom
- Each line is exposed at slightly different time
- Time offset between first and last line

**Advantages:**
- Less expensive
- Better light sensitivity
- Simpler sensor design
- Higher resolutions available at lower cost

**Disadvantages:**
- Moving objects appear distorted ("jello effect")
- Straight lines become diagonal or curved
- Not suitable for motion applications

**When Acceptable:**
- Stationary product inspection
- Product stops before inspection
- Very slow motion (<10mm/s)
- Cost-sensitive applications

**Visual Comparison:**

```
GLOBAL SHUTTER (Moving Object):
┌────────┐
│ ████   │  ← Product appears normal
│ ████   │     (all pixels captured simultaneously)
└────────┘

ROLLING SHUTTER (Moving Object):
┌────────┐
│   ████ │  ← Product appears skewed
│  ████  │     (top captured before bottom)
└────────┘
     Movement direction →
```

**Critical Rule:**
**If your product moves during image capture, you MUST use global shutter.**

**What to Specify:**
- Shutter type required: Global Shutter / Rolling Shutter
- Maximum object motion during exposure: _____ mm
- Stationary vs moving inspection: _____

---

### 5. SENSOR TYPE: CMOS vs CCD

**Definition:** The underlying technology used to convert light into electrical signals.

**CMOS (Complementary Metal-Oxide-Semiconductor):**

**Characteristics:**
- Each pixel has its own amplifier
- Lower power consumption (2-5W typical)
- Faster readout speeds
- Less expensive to manufacture
- Dominant modern technology (>95% of new cameras)

**Advantages:**
- Higher frame rates possible
- Lower cost
- Less heat generation
- Smaller camera size
- Better for high-speed inspection

**Disadvantages:**
- Slightly higher noise levels (improving with each generation)
- Less uniform pixel response

**CCD (Charge-Coupled Device):**

**Characteristics:**
- Charges transferred across chip to single amplifier
- Higher power consumption (5-15W)
- Slower readout
- More expensive
- Legacy technology (being phased out)

**Advantages:**
- Better image uniformity
- Lower noise (historically)
- Better light sensitivity in older designs
- More mature technology

**Disadvantages:**
- Slower frame rates
- Higher cost
- More heat
- Larger size
- Being discontinued by many manufacturers

**Modern Reality:**
**CMOS has largely surpassed CCD in performance. Unless you have specific legacy requirements, choose CMOS.**

**Recommendation Matrix:**

| Application | Sensor Choice | Reason |
|------------|---------------|---------|
| New project | CMOS | Better performance, cost, availability |
| High-speed (>60fps) | CMOS | Faster readout |
| Low light | CMOS (modern) | Latest sensors rival CCD |
| Budget-conscious | CMOS | Lower cost |
| Legacy system match | CCD (if existing) | System compatibility |

**What to Specify:**
- Sensor technology: CMOS (recommended) / CCD (if required)
- Justification if CCD: _____

---

### 6. COLOR vs MONOCHROME (Grayscale)

**Definition:** Whether the sensor captures color information or only brightness (intensity) values.

**MONOCHROME (GRAYSCALE) CAMERAS:**

**How It Works:**
- Each pixel directly measures light intensity
- No color filter array
- Single value per pixel (0-255 for 8-bit)
- All incoming light contributes to measurement

**Advantages:**
- **Higher sensitivity:** 2-3× more light reaching sensor
- **Better resolution:** True resolution (no interpolation)
- **Faster processing:** 1/3 the data size vs color
- **Lower cost:** Simpler sensor design
- **Better contrast:** Enhanced edge detection

**Image Size Comparison:**
```
Monochrome 5MP: 2448 × 2048 × 1 byte = 5 MB
Color 5MP:      2448 × 2048 × 3 bytes = 15 MB
```

**When to Use Monochrome:**
- Defect detection based on shape, texture, or dimension
- High-speed inspection (faster processing)
- Low-light conditions
- Size, position, or presence/absence inspection
- Barcode, OCR, or pattern recognition
- When color information is not needed

**COLOR CAMERAS:**

**How It Works:**
- Bayer filter array over pixels (RGGB pattern)
- Each pixel captures only one color
- Full color image reconstructed through interpolation
- 3 values per pixel (RGB)

**Bayer Pattern:**
```
R  G  R  G  R  G
G  B  G  B  G  B
R  G  R  G  R  G
G  B  G  B  G  B
```

**Advantages:**
- Detect color-based defects (discoloration, staining)
- Distinguish materials by color
- More intuitive for human review
- Can separate defects from colored backgrounds

**Disadvantages:**
- Lower light sensitivity (2/3 of light filtered out)
- Lower effective resolution (interpolation between pixels)
- 3× larger image files
- 3× longer processing time
- More expensive
- Requires color calibration

**When to Use Color:**
- Color is critical defect indicator (rust, discoloration, wrong color)
- Multi-colored products requiring color-based sorting
- Food inspection (ripeness, contamination)
- Print quality inspection
- Situations where color provides critical information

**Decision Tree:**

```
Does defect have different color than good product?
├─ YES → Color camera likely needed
│         (but test with colored lighting on mono first)
└─ NO → Monochrome camera recommended
          (faster, more sensitive, cheaper)
```

**Pro Tip - Alternative to Color Camera:**

Sometimes you can use **monochrome camera + colored lighting**:

- Red light + mono camera = emphasizes red areas
- Blue light + mono camera = emphasizes blue areas
- UV light + mono camera = reveals fluorescence
- Often cheaper and faster than color camera

**Example:**
Detecting brown rust spots on silver metal:
- Option 1: Color camera (expensive, slow)
- Option 2: Monochrome camera + blue LED light (rust appears dark, metal appears bright)

**What to Specify:**
- Camera type: Monochrome / Color
- If color: Required color accuracy (standard / high-precision)
- Lighting wavelength (if monochrome): _____ nm

---

### 7. BIT DEPTH (Dynamic Range)

**Definition:** The number of distinct brightness levels each pixel can represent, determining the camera's ability to capture subtle intensity variations.

**Common Bit Depths:**

| Bit Depth | Gray Levels | Dynamic Range | Typical Use |
|-----------|-------------|---------------|-------------|
| 8-bit | 256 (0-255) | 48 dB | Standard inspection |
| 10-bit | 1,024 | 60 dB | Better detail |
| 12-bit | 4,096 | 72 dB | High-end inspection |
| 14-bit | 16,384 | 84 dB | Scientific imaging |
| 16-bit | 65,536 | 96 dB | Ultra-precision |

**Why It Matters:**

Higher bit depth allows you to:
1. Capture both very bright and very dark areas simultaneously
2. See subtle defects in low-contrast situations
3. Process images without losing information

**Dynamic Range:**
```
Dynamic Range (dB) = 20 × log₁₀(2^Bit Depth)

8-bit:  20 × log₁₀(256)   = 48 dB
12-bit: 20 × log₁₀(4096)  = 72 dB
```

**Practical Example:**

Inspecting shiny metal surface with scratches:
- **8-bit camera:** Bright reflections = 255 (saturated), dark scratches = 50
  - Detail lost in bright areas (all become white)
  
- **12-bit camera:** Bright reflections = 4000, dark scratches = 50
  - Can see texture in bright areas AND detect dark scratches
  - Can digitally adjust contrast later without losing info

**Trade-offs:**

**Higher Bit Depth:**
✓ Better detail in highlights and shadows
✓ More post-processing flexibility
✓ Better for high-contrast scenes
✗ Larger file sizes
✗ Slower processing
✗ More expensive

**When 8-bit is Sufficient:**
- Uniform lighting with controlled contrast
- Simple presence/absence detection
- Well-defined defects (high contrast)
- Speed is critical priority
- Standard inspection applications

**When 10-12 bit is Needed:**
- Reflective or shiny surfaces
- Mixed lighting conditions (bright + shadows)
- Subtle defects (low contrast)
- Need to enhance images digitally
- High-end quality inspection

**What to Specify:**
- Minimum bit depth: _____ bit
- Lighting contrast level: Low / Medium / High
- Post-processing enhancement needed: Yes / No

---

## PAGE 2: OPTICAL & INTERFACE SPECIFICATIONS

### 8. LENS MOUNT TYPE

**Definition:** The mechanical interface between camera body and lens, determining lens compatibility.

**Common Industrial Mounts:**

**C-Mount (Most Common):**
- Thread: 1" diameter, 32 threads per inch (TPI)
- Flange distance: 17.526mm (distance from mount to sensor)
- **Most popular for industrial cameras**
- Wide lens selection available
- Cost-effective

**CS-Mount:**
- Same thread as C-mount (1"-32)
- Flange distance: 12.526mm (5mm shorter)
- Requires CS-mount lenses specifically
- More compact design
- **Can adapt CS-mount camera to C-mount lens with 5mm spacer ring**
- **Cannot use CS-mount lens on C-mount camera (won't focus)**

**M42 Mount:**
- Thread: 42mm × 1mm pitch
- Used in some scientific cameras
- Good for larger sensors

**F-Mount / EF-Mount:**
- DSLR-style mounts
- For large sensor cameras
- High-quality optics available

**Critical Compatibility Rules:**

```
✓ C-mount camera + C-mount lens = Perfect
✓ CS-mount camera + CS-mount lens = Perfect
✓ CS-mount camera + C-mount lens + 5mm spacer = Works
✗ C-mount camera + CS-mount lens = CANNOT FOCUS
```

**What to Specify:**
- Lens mount type: C-Mount / CS-Mount / Other _____
- Sensor size compatibility with lens: _____

---

### 9. LENS SPECIFICATIONS

While the lens is separate from the camera, you must specify camera-lens compatibility requirements:

**Focal Length:**

Determines magnification and field of view:

```
Field of View (mm) = (Sensor Size (mm) × Working Distance (mm)) / Focal Length (mm)

Example:
Sensor: 2/3" (8.8mm wide)
Working distance: 500mm
Focal length: 16mm

FOV = (8.8 × 500) / 16 = 275mm

If you need to inspect 200mm wide product, this lens works.
```

**Common Focal Lengths:**
- 4mm - 8mm: Wide angle, large FOV, close working distance
- 12mm - 16mm: Standard, balanced FOV
- 25mm - 35mm: Telephoto, narrow FOV, distant subjects
- 50mm+: High magnification, small details

**Working Distance:**

Distance from lens front to subject:

```
Minimum Working Distance (WD_min) = Focal Length × (1 + 1/Magnification)

Typical ranges:
- Macro lenses: 50mm - 300mm WD
- Standard lenses: 200mm - 1000mm WD
- Telecentric lenses: Fixed WD (very precise)
```

**F-Number (Aperture):**

Controls depth of field and light gathering:

```
Depth of Field = (2 × F-number × Circle of Confusion × (1 + Magnification)²) / Magnification²

Practical effect:
- f/1.4 (wide open): Shallow DOF, maximum light, less sharp
- f/8 (mid-range): Balanced DOF and sharpness
- f/16 (stopped down): Deep DOF, less light, sharpest
```

**For defect detection:**
- Flat products: f/8 - f/11 (good sharpness, sufficient DOF)
- 3D products with height variation: f/11 - f/16 (deeper DOF)
- Low light: f/2.8 - f/5.6 (more light, but may sacrifice DOF)

**Lens Resolving Power:**

Lens must resolve at least as much detail as sensor:

```
Required Lens Resolution = 1 / (2 × Pixel Size × Magnification)

Example:
Pixel size: 3.45μm
Magnification: 0.1× (100mm FOV on 10mm sensor)
Required lens resolution = 1 / (2 × 3.45μm × 0.1) = 1449 line pairs/mm

Choose lens with ≥ 1500 lp/mm (high-quality lens needed)
```

**What to Specify for Camera-Lens Compatibility:**
- Sensor optical format: _____ inch
- Required field of view: _____ mm × _____ mm
- Working distance: _____ mm
- Depth of field needed: _____ mm
- Lens mount compatibility: C / CS / Other

---

### 10. CAMERA INTERFACE (Data Connection)

**Definition:** The physical connection and protocol used to transfer image data from camera to computer.

**USB 3.0 / USB3 Vision:**

**Specifications:**
- Bandwidth: ~400 MB/s (SuperSpeed)
- Cable length: Maximum 5 meters (without active cables)
- Power: Can power camera via USB (up to 900mA @ 5V)
- Plug-and-play: Easy setup

**Advantages:**
- Simple connection (single cable)
- No special network configuration
- Lower cost
- Good for development/prototyping
- Direct power delivery

**Disadvantages:**
- Short cable length
- Not suitable for industrial environments
- Limited multi-camera support
- Single host computer dependency

**Best For:**
- Desktop/lab setups
- Development and testing
- Single camera systems
- Non-critical applications
- Budget projects

**GigE Vision (Gigabit Ethernet):**

**Specifications:**
- Bandwidth: ~125 MB/s (1000 Mbps / 8)
- Cable length: Up to 100 meters (standard Cat5e/Cat6)
- Can extend to kilometers with fiber optics
- Industry standard protocol (GigE Vision)

**Advantages:**
- Long cable runs (industrial standard)
- Robust CAT6 cables
- Multiple cameras on single network
- Power over Ethernet (PoE) option
- Network switches for distribution
- Remote camera placement

**Disadvantages:**
- Requires network configuration (IP addresses)
- Lower bandwidth than USB 3.0
- Slightly higher latency
- More complex initial setup

**Network Configuration Example:**
```
Camera IP:      192.168.10.50
Subnet Mask:    255.255.255.0
Gateway:        192.168.10.1

Computer NIC:   192.168.10.100 (must be same subnet)

For multiple cameras:
Camera 1: 192.168.10.51
Camera 2: 192.168.10.52
Camera 3: 192.168.10.53
```

**Best For:**
- Production environments
- Multi-camera systems
- Long cable runs (>5m)
- Industrial settings
- Permanent installations

**USB 3.1 / 3.2 (Gen 2):**

- Bandwidth: Up to 1 GB/s
- Cable length: Still limited to ~5m
- Higher cost
- For very high-resolution/high-speed cameras

**Camera Link:**

**Specifications:**
- Bandwidth: 255-680 MB/s (base/medium/full configuration)
- Cable length: 10 meters typical
- Specialized frame grabber required

**Advantages:**
- Very high bandwidth
- Deterministic timing
- Proven in industrial applications
- Low latency

**Disadvantages:**
- Expensive frame grabber card ($500-$2000)
- Proprietary cables
- Short cable length
- Being superseded by newer standards

**Best For:**
- Legacy systems
- Ultra-high-speed inspection
- When maximum bandwidth needed

**CoaXPress (CXP):**

**Specifications:**
- Bandwidth: Up to 12.5 Gb/s (CXP-12)
- Cable length: 40+ meters
- Single coaxial cable
- Power over cable (up to 13W)

**Advantages:**
- Very high bandwidth
- Long cable runs
- Single cable for data + power + triggers
- Future-proof technology

**Disadvantages:**
- Expensive frame grabber
- Higher cost overall
- Newer standard (less mature)

**Best For:**
- High-end inspection systems
- Line scan cameras
- Future installations

**5GigE / 10GigE:**

- Bandwidth: 625 MB/s (5GigE) or 1.25 GB/s (10GigE)
- Requires special network cards
- For ultra-high resolution cameras
- Emerging technology

**Interface Selection Decision Matrix:**

| Need | Recommended Interface |
|------|---------------------|
| Simple setup, short distance (<3m) | USB 3.0 |
| Industrial environment, cable >5m | GigE Vision |
| Multiple cameras, distributed | GigE Vision |
| Maximum bandwidth, cost not issue | CoaXPress or 10GigE |
| Legacy system integration | Camera Link |
| Development/prototyping | USB 3.0 |

**Bandwidth Calculation:**

```
Required Bandwidth = Resolution × Frame Rate × Bit Depth × Color Channels / 8

Example:
5MP (2448×2048), 30fps, 8-bit, mono
= 2448 × 2048 × 30 × 8 × 1 / 8
= 150 MB/s

→ GigE (125 MB/s) insufficient, need USB 3.0 (400 MB/s) or better
   OR reduce to 25 fps to fit GigE
```

**What to Specify:**
- Preferred interface: USB 3.0 / GigE / Camera Link / CoaXPress
- Cable length required: _____ meters
- Number of cameras: _____
- Power over cable needed: Yes / No
- Network infrastructure available: Yes / No

---

### 11. TRIGGERING & SYNCHRONIZATION

**Definition:** Methods to precisely control when the camera captures an image, essential for coordinating with product movement or other systems.

**Trigger Modes:**

**FREE RUNNING (Continuous):**
- Camera captures as fast as possible
- No external synchronization
- Simplest mode

**Use when:**
- Product always in field of view
- Timing not critical
- Development/testing

**SOFTWARE TRIGGER:**
- Computer sends command over data interface
- Camera captures after receiving command
- Timing precision: ~1-10ms (limited by software)

**Use when:**
- Occasional captures
- Timing precision not critical
- Simple setups

**HARDWARE TRIGGER:**
- External electrical signal triggers capture
- Typically 3.3V or 24V digital pulse
- Timing precision: <1μs (microsecond)
- Most common in production

**Signal Types:**
- **Rising edge:** Trigger on 0V→3.3V transition
- **Falling edge:** Trigger on 3.3V→0V transition
- **Level:** Trigger while signal is high/low

**Use when:**
- Product position detection (photoelectric sensor)
- Synchronizing with conveyor encoder
- Multi-camera synchronization
- Production environment (required)

**Trigger Sources:**

1. **Photoelectric Sensor:**
   - Detects product entering inspection zone
   - Through-beam or reflective types
   - Output: 24V or relay contact

2. **Encoder:**
   - Measures conveyor movement
   - Triggers every X millimeters of travel
   - Ensures consistent product position

3. **PLC:**
   - Programmable logic controller
   - Coordinates entire production line
   - Sends trigger pulse when ready

**Trigger Input Specifications:**

Check camera datasheet for:
- **Voltage range:** 3.3V, 5V, 12V, or 24V
- **Input impedance:** Typically >10kΩ
- **Minimum pulse width:** Often 1μs - 1ms
- **Maximum trigger rate:** e.g., 1000 triggers/second
- **Debounce:** Built-in filtering of noise

**Example Trigger Circuit:**

```
                            Camera
Sensor (24V) ─────┬────── Trigger Input (3.3V)
                  │
                 ┴ 
            Voltage Divider or
            Optocoupler Module
```

**Encoder-Based Triggering:**

```
Conveyor moves at 500mm/s
Want image every 100mm of travel
Encoder resolution: 1000 pulses/revolution
Wheel circumference: 500mm

Pulses per 100mm = (1000 pulses/rev) × (100mm / 500mm) = 200 pulses

→ Configure camera to trigger every 200 encoder pulses
```

**STROBE OUTPUT (Flash Synchronization):**

- Camera outputs signal to trigger external flash/strobe
- Freezes motion with short, intense light pulse
- Synchronized precisely with camera exposure

**Timing:**
```
Trigger ─┐     ┌─────────┐
         └─────┘         └─────
         
Exposure    ┌────────┐
            └────────┘
            
Strobe        ┌──┐
              └──┘
              ↑
          Strobe delay (adjustable)
```

**Use when:**
- High-speed moving products
- Insufficient continuous lighting
- Need to freeze motion

**Multi-Camera Synchronization:**

**Master-Slave Configuration:**
```
Master Camera ─── Trigger Out ──┬─── Slave Camera 1
                                └─── Slave Camera 2
                                └─── Slave Camera 3

Master receives external trigger, then triggers all slaves
Ensures all cameras capture at exactly same moment
```

**What to Specify:**
- Trigger type required: Free-run / Software / Hardware
- Trigger input voltage: 3.3V / 5V / 24V / Other _____
- Trigger edge: Rising / Falling / Either
- Minimum trigger rate: _____ fps
- Strobe output required: Yes / No
- Multi-camera sync needed: Yes / No

---

### 12. EXPOSURE CONTROL

**Definition:** The duration for which the sensor collects light, measured in microseconds (μs) or milliseconds (ms).

**Exposure Time Range:**

| Application | Typical Exposure |
|------------|-----------------|
| High-speed moving objects | 10μs - 100μs |
| Standard conveyor | 100μs - 1ms |
| Stationary objects | 1ms - 50ms |
| Low-light conditions | 50ms - 500ms |

**Exposure Calculation for Moving Objects:**

```
Maximum Exposure = Acceptable Motion Blur / Object Velocity

Example:
Object velocity: 1 m/s = 1000 mm/s
Acceptable blur: 0.2mm (less than half smallest defect)

Max exposure = 0.2mm / 1000mm/s = 0.0002s = 200μs
```

**Exposure Modes:**

**MANUAL (Fixed Exposure):**
- You set exact exposure time
- Remains constant
- Most common in industrial inspection

**AUTO EXPOSURE:**
- Camera automatically adjusts
- Tries to achieve target brightness
- Can cause inconsistent results

**For production:** Use manual exposure for consistency

**Exposure Time vs Frame Rate:**

```
Maximum Frame Rate ≤ 1 / (Exposure Time + Readout Time)

Example:
Exposure time: 5ms
Sensor readout time: 15ms
Max FPS = 1 / (5ms + 15ms) = 50 fps
```

**Considerations:**

1. **Motion blur:** Longer exposure = more blur on moving objects
2. **Brightness:** Longer exposure = brighter image (more light)
3. **Frame rate:** Longer exposure = lower max frame rate
4. **Lighting:** Shorter exposure requires more light intensity

**What to Specify:**
- Exposure time range: _____ μs to _____ ms
- Exposure control: Manual / Auto
- Object motion during capture: _____ mm/s
- Maximum acceptable motion blur: _____ mm

---

## PAGE 3: ENVIRONMENTAL & ADVANCED SPECIFICATIONS

### 13. GAIN & SENSITIVITY

**Definition:** Electronic amplification of the sensor signal, increasing brightness of the captured image.

**Gain Specification:**
- Measured in decibels (dB)
- Range: 0 dB (no gain) to 30+ dB (high amplification)
- Alternatively: ISO equivalent (ISO 100, ISO 400, ISO 1600, etc.)

**Gain Formula:**
```
Output Signal = Input Signal × Gain

Gain (dB) = 20 × log₁₀(Amplification Factor)

0 dB = 1× amplification (no gain)
6 dB = 2× amplification
12 dB = 4× amplification
20 dB = 10× amplification
```

**Effect of Gain:**

**Increasing Gain:**
- ✓ Brighter image in low light
- ✓ Can use shorter exposure (less motion blur)
- ✗ Increases noise (grainy image)
- ✗ Reduces effective dynamic range
- ✗ Can hide subtle defects

**Visual Noise Comparison:**
```
0 dB Gain:    ░░░░░░░░ (smooth, low noise)
12 dB Gain:   ▒▒▒▒▒▒▒▒ (moderate noise)
24 dB Gain:   ▓▓▓▓▓▓▓▓ (high noise, grainy)
```

**When to Use Gain:**

Use gain sparingly:
1. **Prefer better lighting** over high gain
2. **Use gain only if:**
   - Lighting already maximized
   - Cannot use longer exposure (motion blur)
   - Temporary solution until lighting improved

**Optimal Strategy:**
```
Priority 1: Optimize lighting (add more lights, better positioning)
Priority 2: Increase exposure time (if object stationary)
Priority 3: Use moderate gain (6-12 dB maximum)
Priority 4: If still dark, upgrade lighting system
```

**Sensor Sensitivity Metrics:**

**Quantum Efficiency (QE):**
- Percentage of photons converted to electrons
- Higher = better light sensitivity
- Modern CMOS: 60-80% QE at peak wavelength

**Spectral Response:**
- Sensitivity varies by light wavelength
- Peak sensitivity: Usually 500-600nm (green light)
- UV sensitivity (<400nm): Often blocked by lens/filter
- IR sensitivity (>700nm): Some sensors extend into near-IR

**Example Spectral Response:**
```
Wavelength (nm)  |  Relative Sensitivity
    400 (UV)     |  30%
    500 (Blue)   |  85%
    550 (Green)  |  100% ← Peak
    650 (Red)    |  70%
    800 (IR)     |  40%
    900 (IR)     |  10%
```

**For defect detection:**
- Match lighting wavelength to sensor peak sensitivity
- If using IR lighting (850nm), ensure camera has good IR response
- Check if IR cut filter is present (removes IR, may need removal)

**Signal-to-Noise Ratio (SNR):**

```
SNR (dB) = 20 × log₁₀(Signal / Noise)

Good SNR values:
- >40 dB: Excellent, clean image
- 30-40 dB: Good
- 20-30 dB: Acceptable
- <20 dB: Poor, noisy

High gain reduces SNR (increases noise faster than signal)
```

**What to Specify:**
- Maximum acceptable gain: _____ dB or ISO _____
- Minimum required SNR: _____ dB
- Spectral sensitivity range: _____ nm to _____ nm
- IR sensitivity needed: Yes / No

---

### 14. TEMPERATURE SPECIFICATIONS

**Definition:** Operating and storage temperature ranges the camera can withstand, and heat generated by camera.

**Operating Temperature Range:**

**Standard Industrial Cameras:**
- **Standard:** 0°C to +45°C (32°F to 113°F)
- **Extended:** -10°C to +60°C (14°F to 140°F)
- **Extreme:** -40°C to +85°C (-40°F to 185°F)

**Storage Temperature:**
- Typically: -20°C to +70°C
- Camera can survive these temps when powered off

**Why Temperature Matters:**

1. **Sensor Dark Current:**
   - Thermal noise increases with temperature
   - Doubles approximately every 8-10°C increase
   - Cold sensors = cleaner images

2. **Electronic Component Reliability:**
   - High temp reduces component lifespan
   - Can cause drift in calibration

3. **Dimensional Stability:**
   - Thermal expansion affects precise measurements
   - Critical for metrology applications

**Camera Heat Generation:**

**Power Consumption Examples:**
- Small USB camera: 2-3W
- Standard GigE camera: 3-5W
- High-resolution / high-speed: 8-15W

**Cooling Requirements:**

**Passive (Fanless):**
- Most industrial cameras use heat sink design
- Requires adequate ventilation
- Ambient temp must stay within spec
- Typical: Up to 5-8W dissipation

**Active (Fan-Cooled):**
- Built-in fan
- Can operate in hotter environments
- Risk: Fan failure = camera overheating
- Requires periodic cleaning

**Thermoelectric Cooling (TEC):**
- Peltier cooler built into camera
- Maintains sensor at constant low temp (e.g., -10°C)
- Benefits: Extremely low noise, stable performance
- Used in: Scientific imaging, low-light inspection
- Cost: $3,000 - $10,000+

**Environmental Enclosures:**

If ambient temperature extreme:

**Heated Enclosure (Cold Environments):**
- Maintains +20°C inside
- Small heater and thermostat
- Prevents condensation

**Cooled Enclosure (Hot Environments):**
- Vortex cooler or AC unit
- Maintains safe operating temp
- Common in foundries, ovens

**Temperature Management Best Practices:**

1. **Ventilation:** Ensure airflow around camera
2. **Mounting:** Don't enclose tightly without cooling
3. **Monitoring:** Log camera temp if available
4. **Alerts:** Set alarm if temp exceeds threshold

**What to Specify:**
- Ambient operating temperature: _____ °C to _____ °C
- Heat generation acceptable: _____ W
- Cooling method: Passive / Active fan / Thermoelectric
- Enclosure needed: Yes / No / Heated / Cooled

---

### 15. INGRESS PROTECTION (IP) RATING

**Definition:** Standardized rating (IP Code) indicating protection against dust and water intrusion.

**IP Rating Format:** IP**XY**
- **X** = Dust protection (0-6)
- **Y** = Water protection (0-9)

**Dust Protection (First Digit):**

| Rating | Protection Level |
|--------|-----------------|
| IP0X | No protection |
| IP1X | >50mm objects (hand) |
| IP2X | >12.5mm objects (finger) |
| IP3X | >2.5mm objects (tools) |
| IP4X | >1mm objects (wires) |
| IP5X | Dust protected (limited ingress) |
| IP6X | Dust tight (no ingress) |

**Water Protection (Second Digit):**

| Rating | Protection Level |
|--------|-----------------|
| IPX0 | No protection |
| IPX1 | Dripping water (vertical) |
| IPX2 | Dripping water (15° tilt) |
| IPX3 | Spraying water (60°) |
| IPX4 | Splashing water (any direction) |
| IPX5 | Water jets (6.3mm nozzle) |
| IPX6 | Powerful water jets (12.5mm) |
| IPX7 | Immersion up to 1m depth |
| IPX8 | Immersion >1m depth |
| IPX9 | High-pressure, high-temp jets |

**Common Industrial Camera Ratings:**

- **IP40:** Basic protection, office/lab use
- **IP54:** Dust protected, splash resistant (standard industrial)
- **IP65:** Dust tight, water jet resistant (good for production)
- **IP67:** Dust tight, temporary immersion (harsh environments)
- **IP69K:** Dust tight, high-pressure/high-temp washdown (food, pharma)

**Application-Based Selection:**

| Environment | Recommended Rating |
|------------|-------------------|
| Clean room / office | IP40 |
| Standard factory floor | IP54 |
| Outdoor / dusty | IP65 |
| Washdown areas | IP67 or IP69K |
| Food processing | IP65 - IP69K |
| Chemical exposure | IP67 + material compatibility |

**Camera IP Rating vs Enclosure:**

**Option 1: IP-Rated Camera**
- Camera itself has IP rating
- More expensive
- Compact
- Limited lens changes

**Option 2: Standard Camera + IP-Rated Enclosure**
- Camera is IP40, but housed in IP65 enclosure
- More flexible
- Can upgrade camera easily
- Enclosure cost: $200-$1000

**Enclosure Considerations:**
- Optical window (glass or acrylic)
- Heated window (prevents condensation)
- Purge air (positive pressure keeps dust out)
- Cooling/heating if needed

**What to Specify:**
- Required IP rating: IP____ 
- Environmental hazards: Dust / Water / Chemicals / Other _____
- Washdown requirements: Yes / No, Frequency: _____
- Enclosure acceptable: Yes / No

---

### 16. POWER REQUIREMENTS

**Definition:** Electrical power needed to operate the camera.

**Common Power Options:**

**USB Powered:**
- Voltage: 5V DC
- Current: 500mA (USB 2.0) to 900mA (USB 3.0)
- Total power: 2.5W - 4.5W
- Convenient for small cameras

**PoE (Power over Ethernet):**
- Voltage: 48V DC (over network cable)
- Standards: 
  - PoE (802.3af): Up to 15.4W
  - PoE+ (802.3at): Up to 30W
  - PoE++ (802.3bt): Up to 90W
- Single cable for data + power
- Common in GigE cameras

**External Power Supply:**
- Voltage: Typically 12V or 24V DC
- Connector: Hirose, M12, barrel jack
- Separate power cable
- Most flexible

**Power Consumption Examples:**

| Camera Type | Typical Power |
|------------|--------------|
| Small USB 3.0, 5MP | 3W |
| GigE, 5MP, standard | 4-6W |
| GigE, 12MP, high-speed | 8-12W |
| Camera Link, high-end | 10-20W |
| TEC-cooled camera | 25-50W |

**Power Supply Sizing:**

Always add 20-30% margin:
```
Required Power Supply = Camera Power × 1.3

Example:
Camera: 6W
Power supply: 6W × 1.3 = 8W minimum

Choose 10W power supply for safety margin
```

**Multi-Camera Power:**
```
Total Power = (Cameras × Power per Camera) × 1.3 + Lighting Power

Example:
3 cameras × 5W = 15W
LED lighting: 10W
Total = (15W + 10W) × 1.3 = 32.5W

Choose 40W power supply
```

**Power Considerations:**

1. **Voltage Regulation:**
   - Industrial cameras sensitive to voltage variation
   - Use regulated supplies (±5% max variation)
   - Avoid switching supplies with high ripple

2. **Grounding:**
   - Proper grounding prevents noise
   - Use isolated power supplies in industrial settings

3. **Backup Power:**
   - UPS (Uninterruptible Power Supply) for critical inspection
   - Prevents data loss during power interruptions

**What to Specify:**
- Power input: USB / PoE / 12V DC / 24V DC / Other _____
- Maximum power consumption: _____ W
- Voltage tolerance: ±_____ %
- UPS/backup power needed: Yes / No

---

### 17. MECHANICAL SPECIFICATIONS

**Definition:** Physical dimensions, weight, and mounting options.

**Camera Dimensions:**

Typical industrial camera sizes:
- **Compact:** 29mm × 29mm × 30mm cube
- **Standard:** 50mm × 50mm × 60mm
- **Large / High-res:** 80mm × 80mm × 100mm

**Weight:**
- Small USB cameras: 50-100g
- Standard industrial: 150-300g
- Large/cooled cameras: 500-1000g

**Why It Matters:**
- Space constraints in tight inspection stations
- Mounting bracket design and strength
- Vibration sensitivity (heavier = more inertia)

**Mounting Options:**

**Tripod Mount:**
- 1/4"-20 threaded hole (standard camera tripod)
- Simple, not robust for production
- Good for prototyping

**C-Mount as Mounting:**
- Some cameras use lens mount for physical mounting
- Not recommended (lens bears weight)
- Can cause misalignment

**Dedicated Mounting Holes:**
- M3 or M4 threaded holes on camera body
- Standard pattern (e.g., 4 holes on 40mm spacing)
- Robust, industrial standard

**DIN Rail Mount:**
- Clips onto standard DIN rail
- Common in control panels
- Easy installation/removal

**Mounting Bracket Design:**

For stable imaging:
1. **Rigid mounting:** Minimize vibration
2. **Adjustable:** X-Y-Z positioning
3. **Repeatable:** Locked position doesn't drift
4. **Accessible:** Can adjust focus/aperture

**Cable Management:**

**Cable Types:**
- Data cable (USB, Ethernet)
- Power cable (if separate)
- Trigger input cable
- Strobe output cable

**Cable Protection:**
- Strain relief at camera connector
- Cable chain for moving cameras
- Shielded cables in high EMI environments
- Proper bend radius (typically >10× cable diameter)

**What to Specify:**
- Maximum camera dimensions: _____ mm × _____ mm × _____ mm
- Maximum weight: _____ g
- Mounting method: Tripod / Bracket / DIN Rail / Custom
- Cable length required: _____ m
- Cable flexibility: Fixed / Moving (cable chain)

---

### 18. EMC (Electromagnetic Compatibility)

**Definition:** Camera's ability to operate in electromagnetic environment without malfunction or causing interference.

**EMC Standards:**

**Emission (Camera shouldn't emit excessive EM radiation):**
- **FCC Part 15 Class A:** Industrial environments
- **FCC Part 15 Class B:** Residential (stricter)
- **CE / EN 55032:** European standard

**Immunity (Camera should resist external EM interference):**
- **IEC 61000-4-2:** Electrostatic discharge (ESD)
- **IEC 61000-4-3:** Radiated RF immunity
- **IEC 61000-4-4:** Electrical fast transient (burst)
- **IEC 61000-4-5:** Surge immunity

**Common Industrial Interference Sources:**

1. **Motor drives / VFDs:** High-frequency switching noise
2. **Welding equipment:** Intense EM fields
3. **RF transmitters:** Wireless networks, radios
4. **Switching power supplies:** High-frequency noise
5. **Relay coils:** Inductive spikes

**Symptoms of EMI Problems:**

- Random image artifacts (white specks, lines)
- Intermittent camera disconnections
- Corrupted data
- Reduced frame rate
- Complete failure to operate

**EMI Mitigation Strategies:**

**Cabling:**
- **Shielded cables:** Use CAT6a STP (shielded twisted pair) for GigE
- **Proper grounding:** 360° shield grounding at both ends
- **Separate power and signal:** Keep apart, cross at 90° if necessary
- **Ferrite cores:** Add to cables near noise sources

**Mounting:**
- Metal enclosure around camera (grounded)
- Distance from EM noise sources (>0.5m minimum)
- Ground camera chassis

**Power:**
- Filtered power supply
- Isolation transformers
- Dedicated clean power circuit

**Camera Selection:**
- Industrial-grade cameras have better EMC design
- Metal housing (not plastic)
- Integrated EMI filters

**Testing EMC:**

Before production deployment:
1. Run camera in actual production environment
2. Operate nearby machinery (motors, welders, etc.)
3. Monitor for image quality degradation
4. Check error logs for disconnections
5. Test during worst-case scenarios (multiple machines running)

**What to Specify:**
- EMC compliance required: CE / FCC / Other _____
- EM environment: Clean / Moderate / Harsh
- Nearby interference sources: _____
- Shielded cables required: Yes / No

---

### 19. CERTIFICATIONS & STANDARDS

**Definition:** Official certifications confirming camera meets specific safety, quality, or industry standards.

**Common Certifications:**

**CE (European Conformity):**
- Required for sale in European Union
- Covers safety, health, environmental protection
- Low voltage directive, EMC directive
- Self-certified by manufacturer

**FCC (Federal Communications Commission):**
- Required for sale in USA
- Limits electromagnetic interference
- Class A (industrial) or Class B (consumer)

**RoHS (Restriction of Hazardous Substances):**
- Limits lead, mercury, cadmium, etc.
- Important for environmental compliance
- EU and many other regions

**UL (Underwriters Laboratories):**
- Independent safety certification
- UL 60950-1 (IT equipment)
- Not always required but increases confidence

**IEC Standards:**
- **IEC 60950-1:** Safety of IT equipment
- **IEC 61000:** EMC standards
- **IEC 62368-1:** Audio/video equipment safety

**Industry-Specific:**

**FDA (Food & Drug Administration):**
- For medical device manufacturing
- 21 CFR Part 11 compliance if applicable

**ATEX / IECEx:**
- Explosive atmosphere certification
- Required in chemical, oil & gas, mining
- Very specialized, expensive

**IP Rating (Discussed earlier):**
- IEC 60529 standard

**GenICam / GigE Vision:**
- Software interface standards
- Ensures interoperability between cameras and software
- Most industrial cameras comply

**What to Specify:**
- Required certifications: CE / FCC / UL / RoHS / Other _____
- Industry-specific needs: Medical / Food / Hazardous area
- Geographic region of use: _____

---

### 20. LONGEVITY & SUPPORT

**Definition:** Manufacturer's commitment to product availability and support over time.

**Product Lifecycle:**

**Consumer Cameras:**
- Discontinued after 1-2 years
- No replacement parts
- Firmware/driver support ends
- NOT suitable for production

**Industrial Cameras:**
- **Minimum 5-7 year availability** from announcement
- **10+ years** for critical applications
- Advance notice of discontinuation (6-12 months)
- Migration path to replacement model

**Why Long-Term Availability Matters:**

Scenario: You design system around camera, go into production.
- Year 3: Camera fails, need replacement
- Consumer camera: Discontinued, must redesign entire system
- Industrial camera: Still available, order replacement, swap in 30 minutes

**Firmware and Driver Support:**

- **Ongoing updates:** Bug fixes, performance improvements
- **OS compatibility:** Support for new Windows/Linux versions
- **SDK updates:** Compatibility with new development tools

**Technical Support:**

**Levels:**
- **Basic:** Email support, response in 24-48 hours
- **Premium:** Phone support, response in 4-8 hours
- **On-site:** Technician visit for troubleshooting
- **Custom engineering:** Modification for specific needs

**Documentation:**

Quality manufacturers provide:
- Detailed datasheet (all specifications)
- User manual (installation, operation)
- SDK documentation and examples
- Application notes
- 3D CAD models for mechanical design

**Spare Parts:**

- Availability of:
  - Replacement sensors
  - Interface boards
  - Cables and connectors
- RMA (Return Merchandise Authorization) process
- Repair service (in-warranty and out-of-warranty)

**Total Cost of Ownership:**

```
TCO = Purchase Price + Installation + Training + Maintenance + Downtime Cost

Lower-cost camera with poor support:
$500 camera + $5000 downtime (when fails, no replacement) = $5500 TCO

Higher-cost industrial camera with support:
$1500 camera + $0 downtime (rapid replacement) = $1500 TCO

Industrial camera is cheaper long-term!
```

**Vendor Selection Criteria:**

1. **Established manufacturer:** In business 10+ years
2. **Geographic presence:** Local support/distributor
3. **Application engineering:** Help with camera selection
4. **Customer references:** Other successful projects
5. **Financial stability:** Not likely to go out of business

**What to Specify:**
- Minimum product availability: _____ years
- Required support level: Email / Phone / On-site
- Documentation language: _____
- Local support required: Yes / No
- Spare camera inventory: _____ units

---

## SUMMARY CHECKLIST: Camera Specification Template

Use this checklist when specifying a camera for defect detection:

### **OPTICAL REQUIREMENTS**
- [ ] Required field of view: _____ mm × _____ mm
- [ ] Smallest defect size: _____ mm
- [ ] Working distance: _____ mm
- [ ] Depth of field required: _____ mm
- [ ] Calculated resolution needed: _____ MP
- [ ] Selected resolution: _____ MP (_____ × _____ pixels)
- [ ] Color required: Yes / No (Mono recommended if No)

### **PERFORMANCE REQUIREMENTS**
- [ ] Production rate: _____ products/minute
- [ ] Required frame rate: _____ fps
- [ ] Object velocity: _____ mm/s (if moving)
- [ ] Shutter type: Global (required if moving) / Rolling (if stationary)
- [ ] Maximum exposure time: _____ μs or ms
- [ ] Sensor type: CMOS (recommended) / CCD (legacy)

### **INTERFACE & CONNECTIVITY**
- [ ] Interface: USB 3.0 / GigE / Camera Link / CoaXPress
- [ ] Cable length: _____ m
- [ ] Trigger method: Free-run / Software / Hardware
- [ ] Trigger input voltage: 3.3V / 5V / 24V
- [ ] Number of cameras: _____

### **ENVIRONMENTAL**
- [ ] Operating temperature: _____ °C to _____ °C
- [ ] IP rating required: IP_____
- [ ] Dusty environment: Yes / No
- [ ] Water exposure: Yes / No
- [ ] Vibration present: Yes / No

### **ELECTRICAL**
- [ ] Power: USB / PoE / 12V / 24V
- [ ] Maximum power consumption: _____ W
- [ ] EMC environment: Clean / Moderate / Harsh

### **MECHANICAL**
- [ ] Maximum dimensions: _____ × _____ × _____ mm
- [ ] Maximum weight: _____ g
- [ ] Mounting: Bracket / DIN rail / Custom
- [ ] Lens mount: C-mount / CS-mount / Other

### **QUALITY & SUPPORT**
- [ ] Bit depth: 8-bit / 10-bit / 12-bit
- [ ] Product lifecycle: Minimum _____ years
- [ ] Required certifications: CE / FCC / UL / RoHS
- [ ] Warranty: _____ years
- [ ] Support level: Email / Phone / On-site

### **BUDGET**
- [ ] Camera budget: $ _____ per unit
- [ ] Total cameras needed: _____
- [ ] Budget includes lens: Yes / No
- [ ] Budget includes lighting: Yes / No

---

## RECOMMENDED CAMERA SPECIFICATIONS FOR COMMON SCENARIOS

### **SCENARIO 1: Small Parts on Conveyor (Fast Inspection)**

**Application:** Inspecting 50mm × 50mm electronic components at 120 parts/minute

**Specifications:**
- Resolution: 5MP (2448 × 2048)
- Frame rate: 60 fps minimum
- Shutter: Global shutter (required)
- Sensor: CMOS
- Color: Monochrome (faster, more sensitive)
- Interface: GigE (allows remote mounting)
- Trigger: Hardware (encoder or sensor)
- Exposure: 100-500 μs (short to freeze motion)
- Lens: 16mm, f/5.6
- Working distance: 400mm
- IP rating: IP54 minimum

**Example Cameras:**
- Basler ace acA2440-75gm (GigE, 2.3MP, 75fps, global shutter)
- FLIR Blackfly S BFS-U3-50S5M-C (USB3, 5MP, 75fps)

---

### **SCENARIO 2: Large Surface Inspection (High Detail)**

**Application:** Inspecting 300mm × 300mm painted panels for scratches and defects

**Specifications:**
- Resolution: 12-20MP (for fine scratch detection)
- Frame rate: 15-30 fps (stationary inspection)
- Shutter: Global or rolling acceptable (if stationary)
- Sensor: CMOS
- Color: Monochrome or color (depending on defect type)
- Interface: GigE or 5GigE (high bandwidth needed)
- Trigger: Software or hardware
- Exposure: 5-20 ms (higher detail, stationary)
- Lens: 35mm, f/8 (for good depth of field)
- Working distance: 800mm
- IP rating: IP54

**Example Cameras:**
- Basler ace acA4112-20um (USB3, 12MP, 20fps)
- FLIR Blackfly S BFS-U3-200S6C-C (USB3, 20MP, color)

---

### **SCENARIO 3: 3D / High-Speed (Line Scan Camera)**

**Application:** Continuous web inspection (paper, textiles, metals) at 3 m/min

**Specifications:**
- Type: Line scan camera (not area scan)
- Resolution: 4K - 8K pixels per line
- Line rate: 20-100 kHz
- Shutter: Global (inherent in line scan)
- Sensor: CMOS line sensor
- Color: Tri-linear (RGB) or mono
- Interface: Camera Link or CoaXPress (very high bandwidth)
- Trigger: Encoder (exact line position)
- Lighting: Bright LED line light
- Working distance: 200-500mm

**Example Cameras:**
- Basler racer raL4096-24gm (4K line, 24kHz, mono)
- Teledyne DALSA Piranha4 (8K line, color)

---

This guide provides all the technical knowledge you need to properly specify and select a camera for your defect detection project. Use the checklist to ensure you've considered all critical parameters, and match your application to one of the common scenarios for a starting reference.

**Good luck with your camera selection!**
