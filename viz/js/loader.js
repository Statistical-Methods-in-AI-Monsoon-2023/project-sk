import { world } from './world.js'
import * as THREE from 'three'
// import { FBXLoader } from 'three/examples/jsm/loaders/FBXLoader'
// import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader'
// import { MTLLoader } from 'three/examples/jsm/loaders/MTLLoader'
// import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader'

const { textures, models } = world

function load_textures() {

}


function load_models() {

}

async function load_assets() {
	load_textures()
	load_models()
}

export { load_assets }