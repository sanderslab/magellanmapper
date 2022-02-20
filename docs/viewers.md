# Image Viewers in MagellanMapper

The tabbed pane on the right of the graphical interface provides several  image viewers to visualize volumetric images.

## Region of Interest (ROI) Editor

The multi-level 2D plotter is geared toward simplifying annotation for nuclei or other objects. Select the `ROI Editor` tab to view the editor. Press the `Redraw` button to redraw the editor at the selected ROI. To detect and display nuclei in the ROI, select the `Detect` tab and press the `Detect` button.

### Navigation

| To Do...        | Shortcut            |
| ---------------- | :------------------: |
| Increase/decrease the overview plots' z-plane | Arrow `up/right` to increase and `down/left` to decrease |
| Jump to a z-plane in the overview plots corresponding to an ROI plane | `Right-click` on the the corresponding ROI plane |
| Preview the ROI at a certain position | `Left-click` in the overview plot |
| Redraw the editor at the chosen ROI settings | Double `right-click` in any overview plot |

### Annotations

| To Do...        | Shortcut            |
| ---------------- | :------------------: |
| Add a circle | `Ctrl+click` at the desired location |
| Move the circle's position | `shift+click` and drag (note that the original position will remain as a solid circle) |
| Resize the circle's radius | `Alt+click` (`option+click` on Mac) and drag |
| Copy the circle | `"c"+click` |
| Duplicate a circle to the same position in another z-plane | `"v"+click` on the corresponding position in the z-plane to which the circle will be duplicated |
| Cut the circle | `"x"+click` |
| Delete the circle | `"d"+click` |
| Cycle between the 3 nuclei detection flags | Click within the dotted circles; incorrect (red), correct (green), or questionable (yellow) |


## Atlas Editor

The multi-planar image plotter allows simplified viewing and editing of annotation labels for an atlas. Existing labels can be painted into adjacent areas, and synchronized planar viewing allows realtime visualization of changes in each plane.

To view the editor, select the `Atlas Editor` tab. The `Redraw` button in the `ROI` tab of the left panel will redraw the editor if necessary. The `Registered images` section allows selecting any available annotations and label reference files to overlay.

| To Do...       | Shortcut                    |
| ------------- |:-------------------------: |
| See the region name | Mouseover over any label |
| Move the crosshairs and the corresponding planes | `Left-click` |
| Move planes in the current plot | Scroll or arrow `up`/`down`|
| Zoom | `Right-click` or `Ctrl+left-click` while moving the mouse up/down |
| Pan | `Middle-click` or `Shift+left-click` while dragging the mouse |
| Toggle between 0 and full labels alpha (opacity) | `a` |
| Halve alpha | `shift+a` |
| Return to original alpha | press `a` twice |

Press on the "Edit" button to start painting labels using these controls:

| To Do...        | Shortcut                    |
| ------------- |:-------------------------:|
|Paint over a new area | `Left-click`, pick a color, then drag over image |
| Use the last picked color to paint over a new area | `Alt+Left-click`(option-click on Mac) |
| Make the paintbrush smaller/bigger | `[`/`]` (brackets) |
| Halve the increment of the paintbrush size | `[`/`]` and add `shift` |


Use the save button in the main window with the atlas window still open to resave


## 3D viewer

The 3D viewer displays regions of interest, atlas structures, or whole images. MagellanMapper uses the Mayavi data visualizer for 3D voxel or surface rendering.

TODO: add further description
