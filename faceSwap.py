import cv2 as cv
import numpy as np
import dlib
import time
from PySHull import PySHull

def shape_to_np(shape):
    coords = np.zeros((68, 2), dtype=int) # 68 is all array and 2 is 2-tuple of (x,y)-coordinates
    # to a 2-tuple of (x, y)-coordinates
    
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    #print("==",coords)
    return coords


def draw(img ,triList, points):
	arr1 = points[triList,0]
	arr2 = points[triList,1]
	pts1 = (arr1[0],arr2[0])
	pts2 = (arr1[1],arr2[1])
	pts3 = (arr1[2],arr2[2])
	pts4 = (arr1[3],arr2[3])
	cv.line(img , pts1, pts2, (255, 255, 255), 1)
	cv.line(img , pts2, pts3, (255, 255, 255), 1)
	cv.line(img , pts3, pts4, (255, 255, 255), 1)

def barycentric(triangles, pts):
	ct = []
	pt = pts[triangles]
	ln = len(pt)
	for p in pt:
		bary = calArea(p)
		ct.append(bary)
	return ct


def calArea(pt):
	aX = pt[0][0]
	aY = pt[0][1]
	bX = pt[1][0]
	bY = pt[1][1]
	cX = pt[2][0]
	cY = pt[2][1]
	
	pX = (aX + bX + cX) / 3
	pY = (aY + bY + bY) / 3
	#computing area triangle ABC
	BminusA = [(bX - aX), (bY - aY)]
	CminusA = [(cX - aX), (cY - aY)]
	areaTriangle = float(np.cross(BminusA, CminusA))
	

	#computing area triangle CAP
	AminusC = [(aX - cX), (aY - cY)]
	PminusC = [(pX - cX), (pY - aY)]
	areaTriangleCAP = float(np.cross(AminusC, PminusC))
	beta = areaTriangleCAP / areaTriangle
	#computing area triangle ABP
	PminusA = [(pX - aX), (pY - aY)]
	areaTriangleABP = float(np.cross(BminusA, PminusA))
	gamma = areaTriangleABP / areaTriangle
	#computing area triangle BCP
	alpha = 1 - beta - gamma
	
	return (alpha, beta, gamma)

def markpoint(p):
	for (xx, yy) in p:
		cv.circle(img1, (xx, yy), 1, (255, 0, 0), 2)

def readPoints() :
    points =[[637, 327],[645, 370],[653, 411],[663, 454],[681, 492],[710, 524],[748, 546],[790, 564],[837, 565],[880, 553],[917, 529],
			[947, 500],[968, 463],[979, 422],[979, 377],[975, 332],[970, 291],[658, 280],[672, 254],[700, 245],[730, 249],[757, 261],
			[826, 257],[851, 240],[880, 230],[909, 233],[928, 253],[796, 292],[799, 320],[801, 349],[804, 378],[773, 404],[790, 407],
			[808, 410],[825, 403],[841, 396],[698, 309],[713, 300],[730, 301],[749, 309],[731, 313],[713, 314],[845, 300],[859, 289],
			[876, 287],[894, 291],[879, 297],[862, 300],[744, 465],[771, 453],[794, 447],[812, 446],[831, 440],[857, 439],[885, 441],
			[865, 457],[841, 468],[821, 474],[802, 476],[775, 475],[754, 464],[795, 456],[814, 455],[833, 449],[876, 442],[838, 457],
			[818, 463],[799, 465]]
			
    return points
def ListTri(triangle, points):
	tri = []
	for t in triangle:
		dt = points[t[0]],points[t[1]],points[t[2]]
		tri.append(dt)
	return tri

def FinePointsTri(tri, bary):
	aX = tri[0][0]
	aY = tri[0][1]
	bX = tri[1][0]
	bY = tri[1][1]
	cX = tri[2][0]
	cY = tri[2][1]

	alpha = bary[0]
	beta = bary[1]
	gamma = bary[2]

	x = ((alpha*aX)+(beta*bX)+(gamma*cX))
	y = ((alpha*aY)+(beta*bY)+(gamma*cY))
	p = (x, y)
	return p
def warpTriangle(img1, img2, t1, t2) :

    # Find bounding rectangle for each triangle
    r1 = cv.boundingRect(np.float32([t1]))
    r2 = cv.boundingRect(np.float32([t2]))
	
    # Offset points by left top corner of the respective rectangles
    t1Rect = [] 
    t2Rect = []
    t2RectInt = []

    for i in xrange(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))
        t2RectInt.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))
	
    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
    cv.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0);
    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    #img2Rect = np.zeros((r2[3], r2[2]), dtype = img1Rect.dtype)
    
    size = (r2[2], r2[3])
    
    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)
    
    img2Rect = img2Rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask )
     
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Rect 
    
def applyAffineTransform(src, srcTri, dstTri, size) :
    
    # Given a pair of triangles, find the affine transform.
    warpMat = cv.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    
    # Apply the Affine Transform just found to the src image
    dst = cv.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT_101 )
    
    return dst

if __name__ == "__main__":
	startTime = time.time()
	face_cascade = cv.CascadeClassifier('cascade/data/haarcascade_frontalface_alt2.xml')
	PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
	detector = dlib.get_frontal_face_detector()
	predector = dlib.shape_predictor(PREDICTOR_PATH)
	img1 = cv.imread('lungTu.jpg')
	img2 = cv.imread('Donald_Trump.jpg')
	img1Warped = np.copy(img1)
	lungTu = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
	Donald_Trump = cv.cvtColor(img2, cv.COLOR_BGR2GRAY) 
	faces = face_cascade.detectMultiScale(lungTu, scaleFactor=1.3, minNeighbors=7)
	
	x = faces[0][0]
	y = faces[0][1]
	w = faces[0][2]
	h = faces[0][3]
	#cv.rectangle(img1,(x,y),(x+w,y+h),(255,0,0),2)
	roi_gray = lungTu[y:y+h, x:x+w]
	roi_color = img1[y:y+h, x:x+w]
	rects = detector(lungTu, 0)
	
	points2 = readPoints()
	for rect in rects:
		shape = predector(lungTu, rect) 
		shape = shape_to_np(shape)
		points1 = shape   
		dt, hull = PySHull(points1)	                 
		#dt = DelaunayTri(points1)
		#print points1
		"""
		for tri in dt:
			tri1 = list(tri[:])
			tri1.append(tri[0])
			draw(img1, tri1, points1)
		
		markpoint(points1)
		"""
		bary = barycentric(dt,points1)
		dt1 = ListTri(dt, points2)
		num = len(dt1)
		PointsTri = []
		for i in range(num):
			p = FinePointsTri(dt1[i], bary[i])
			PointsTri.append(p)
		
		pt = points1[dt]
		for i in range(num):
			warpTriangle(img2, img1Warped, dt1[i], pt[i])

	print(time.time() - startTime, "sec")
	cv.imshow('Output',output)
	cv.waitKey(0)
	cv.destroyAllWindows()
	