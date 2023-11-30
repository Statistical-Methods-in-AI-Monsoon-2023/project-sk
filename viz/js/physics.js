import * as THREE from 'three'
import { world } from './world.js'
import AmmoLib from "ammo-es"

let Ammo

async function init_physics() {
	await AmmoLib().then(Ammo => {
		world.Ammo = Ammo
	})

	Ammo = world.Ammo

	init()
}


function init() {
	// Heightfield parameters
	world.physics.terrainWidthExtents = 1;
	world.physics.terrainDepthExtents = 1;
	world.physics.terrainWidth = 20;
	world.physics.terrainDepth = 20;
	world.physics.terrainHalfWidth = world.physics.terrainWidth / 2;
	world.physics.terrainHalfDepth = world.physics.terrainDepth / 2;
	world.physics.terrainMaxHeight = 1;
	world.physics.terrainMinHeight = -1;

	// Graphics variables
	world.physics.terrainMesh;

	// Physics variables
	world.physics.collisionConfiguration;
	world.physics.dispatcher;
	world.physics.broadphase;
	world.physics.solver;
	world.physics.physicsWorld;
	world.physics.dynamicObjects = [];
	world.physics.transformAux1;

	world.physics.heightData = null;
	world.physics.ammoHeightData = null;

	// get data from blend_plot
	const geometry = world.blend_plot.processed.mesh.geometry
	if (!geometry) return

	world.physics.heightData = new Float32Array(geometry.vertices.map(v => v.z))

	initGraphics();

	initPhysics();
}

function initGraphics() {
	const { terrainWidthExtents, terrainDepthExtents, terrainWidth, terrainDepth, heightData } = world.physics
	const geometry = new THREE.PlaneGeometry(terrainWidthExtents, terrainDepthExtents, terrainWidth - 1, terrainDepth - 1);
	geometry.rotateX(-Math.PI / 2);

	const vertices = geometry.vertices;

	for (let i = 0; i < vertices.length; i++) {
		vertices[i].y = heightData[i]
	}

	geometry.computeVertexNormals();
	geometry.computeFaceNormals();

	const groundMaterial = new THREE.MeshStandardMaterial({
		color: 'green',
		wireframe: true
	});
	const terrainMesh = new THREE.Mesh(geometry, groundMaterial);

	// double sided
	terrainMesh.material.side = THREE.DoubleSide;
	terrainMesh.visible = false;

	world.scene.add(terrainMesh);

	// world.physics.terrainMesh = world.blend_plot.processed.mesh
	world.physics.terrainMesh = terrainMesh
}

function initPhysics() {

	// Physics configuration

	world.physics.collisionConfiguration = new Ammo.btDefaultCollisionConfiguration();
	world.physics.dispatcher = new Ammo.btCollisionDispatcher(world.physics.collisionConfiguration);
	world.physics.broadphase = new Ammo.btDbvtBroadphase();
	world.physics.solver = new Ammo.btSequentialImpulseConstraintSolver();
	world.physics.physicsWorld = new Ammo.btDiscreteDynamicsWorld(world.physics.dispatcher, world.physics.broadphase, world.physics.solver, world.physics.collisionConfiguration);
	world.physics.physicsWorld.setGravity(new Ammo.btVector3(0, -2, 0));

	// Create the terrain body

	world.physics.groundShape = createTerrainShape();
	world.physics.groundTransform = new Ammo.btTransform();
	const { groundShape, groundTransform, physicsWorld, terrainMaxHeight, terrainMinHeight } = world.physics
	groundTransform.setIdentity();
	// Shifts the terrain, since bullet re-centers it on its bounding box.
	groundTransform.setOrigin(new Ammo.btVector3(0, (terrainMaxHeight + terrainMinHeight) / 2, 0));
	world.physics.groundMass = 0;
	world.physics.groundLocalInertia = new Ammo.btVector3(0, 0, 0);
	world.physics.groundMotionState = new Ammo.btDefaultMotionState(groundTransform);
	const { groundMass, groundLocalInertia, groundMotionState } = world.physics
	world.physics.groundBody = new Ammo.btRigidBody(new Ammo.btRigidBodyConstructionInfo(groundMass, groundMotionState, groundShape, groundLocalInertia));
	const { groundBody } = world.physics
	physicsWorld.addRigidBody(groundBody);

	world.physics.transformAux1 = new Ammo.btTransform();
	world.physics.initialized = true;
}

function createTerrainShape() {
	const heightScale = 1;

	const upAxis = 1;

	const hdt = 'PHY_FLOAT';

	const flipQuadEdges = false;

	const { terrainWidthExtents, terrainDepthExtents, terrainWidth, terrainDepth, terrainHalfWidth, terrainHalfDepth, terrainMaxHeight, terrainMinHeight, heightData } = world.physics

	world.physics.ammoHeightData = Ammo._malloc(4 * terrainWidth * terrainDepth);

	const ammoHeightData = world.physics.ammoHeightData

	let p = 0;
	let p2 = 0;

	for (let j = 0; j < terrainDepth; j++) {
		for (let i = 0; i < terrainWidth; i++) {
			// write 32-bit float data to memory
			Ammo.HEAPF32[ammoHeightData + p2 >> 2] = heightData[p];
			p++;
			// 4 bytes/float
			p2 += 4;
		}
	}

	// Creates the heightfield physics shape
	const heightFieldShape = new Ammo.btHeightfieldTerrainShape(
		terrainWidth,
		terrainDepth,
		ammoHeightData,
		heightScale,
		terrainMinHeight,
		terrainMaxHeight,
		upAxis,
		hdt,
		flipQuadEdges
	);

	// Set horizontal scale
	const scaleX = terrainWidthExtents / (terrainWidth - 1);
	const scaleZ = terrainDepthExtents / (terrainDepth - 1);
	heightFieldShape.setLocalScaling(new Ammo.btVector3(scaleX, 1, scaleZ));

	heightFieldShape.setMargin(0.05);

	return heightFieldShape;

}

function add_ball() {
	// Sphere
	const margin = 0.05;
	const radius = 0.02
	const material = new THREE.MeshStandardMaterial({ color: 'gold', metalness: 0.3, roughness: 0.4 });
	const ball = new THREE.Mesh(new THREE.SphereGeometry(radius, 20, 20), material);
	const shape = new Ammo.btSphereShape(radius);
	shape.setMargin(margin);

	const { dynamicObjects, physicsWorld } = world.physics

	// set position between 0 and 1
	ball.position.x = (Math.random() - 0.5)
	ball.position.y = 1
	ball.position.z = (Math.random() - 0.5)

	const mass = 5;
	const transform = new Ammo.btTransform();
	transform.setIdentity();
	transform.setOrigin(new Ammo.btVector3(...ball.position.toArray()));
	const motionState = new Ammo.btDefaultMotionState(transform);
	const localInertia = new Ammo.btVector3(0, 0, 0);
	shape.calculateLocalInertia(mass, localInertia);
	const rbInfo = new Ammo.btRigidBodyConstructionInfo(mass, motionState, shape, localInertia);
	const body = new Ammo.btRigidBody(rbInfo);

	ball.userData.physicsBody = body;

	ball.receiveShadow = true;
	ball.castShadow = true;

	const { scene } = world
	scene.add(ball);
	dynamicObjects.push(ball);
	physicsWorld.addRigidBody(body);
}

function update_physics(delta) {
	if (!world.physics.initialized) return
	const { physicsWorld, transformAux1, dynamicObjects } = world.physics

	if (dynamicObjects.length === 0) return
	physicsWorld.stepSimulation(delta, 10);

	// Update objects
	for (let i = 0, il = dynamicObjects.length; i < il; i++) {
		const objThree = dynamicObjects[i];
		const objPhys = objThree.userData.physicsBody;
		const ms = objPhys.getMotionState();
		if (ms) {
			ms.getWorldTransform(transformAux1);
			const p = transformAux1.getOrigin();
			const q = transformAux1.getRotation();
			objThree.position.set(p.x(), p.y(), p.z());
			objThree.quaternion.set(q.x(), q.y(), q.z(), q.w());
		}

	}

}

function clear_balls() {
	if (!world.physics.initialized) return

	// Remove all physics objects
	for (let i = 0; i < world.physics.dynamicObjects.length; i++) {
		Ammo.destroy(world.physics.dynamicObjects[i].userData.physicsBody)
		world.physics.physicsWorld.removeRigidBody(world.physics.dynamicObjects[i].userData.physicsBody)
		world.physics.dynamicObjects[i].userData.physicsBody = null
		world.scene.remove(world.physics.dynamicObjects[i])
		world.physics.dynamicObjects[i] = null
	}

	world.physics.dynamicObjects = []
}


function destroy_physics() {
	if (!world.physics.initialized) return

	clear_balls()

	world.scene.remove(world.physics.terrainMesh)

	Ammo = null
}



export { update_physics, init_physics, destroy_physics, add_ball, clear_balls }