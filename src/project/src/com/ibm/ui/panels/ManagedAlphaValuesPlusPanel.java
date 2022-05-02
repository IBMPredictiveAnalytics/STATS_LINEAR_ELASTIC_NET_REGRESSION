/************************************************************************
 ** IBM Confidential
 **
 ** OCO Source Materials
 **
 ** IBM SPSS Products: <Analytic Components>
 **
 ** (C) Copyright IBM Corp. 2009, 2022
 **
 ** The source code for this program is not published or otherwise divested of its trade secrets,
 ** irrespective of what has been deposited with the U.S. Copyright Office.
 ************************************************************************/

package com.ibm.ui.panels;

import com.pasw.framework.common.core.CFEnum;
import com.pasw.framework.common.core.Properties;
import com.pasw.framework.common.session.Session;
import com.pasw.framework.ui.common.UISession;
import com.pasw.framework.ui.swing.ExtensionObjectSwingUI;
import com.pasw.framework.ui.swing.ManagedPanelContext;
import com.pasw.framework.ui.swing.spi.ManagedPanel;
import com.pasw.framework.ui.swing.spi.ManagedUIElement;
import com.pasw.ui.common.control.Control;
import com.pasw.ui.common.control.ControlEvent;
import com.pasw.ui.common.control.ControlListener;
import com.pasw.ui.common.control.ControlManager;
import com.pasw.ui.swing.UIUtilities;
import com.pasw.ui.swing.common.AlertOptionPane;
import com.pasw.ui.swing.common.SwingFeatureUI;
import com.spss.java_client.ui.cf_integration.instance.CFSession;
import com.spss.uitools.res.UIToolResUtil;

import javax.swing.*;
import java.math.BigDecimal;
import java.text.DecimalFormatSymbols;
import java.util.*;

/**
 * Title:       ManagedAlphaValuesPlusPanel
 * Description: A managed panel that contains Mode combo, and Two sets of specify checkboxes,singleValues, start, end, and by numeric fields,
 * One is L1 ratio related, the other one is not
 * Various validation includes:
 * A. For single L1 ratio/alpha single values:
 *  1. make sure one or more numeric values separated by space, and no duplication within the values
 *  2. When Mode=Fit, only one value is permitted.
 *  3. When Specify Grid check is enabled and checked, no duplication against Grid
 * B. For Specify checkboxes: Either individual or grid must be selected
 * C. For Grid: Start/End/By values:
 *  1. make sure no duplication in Single values against grid specification
 *  2. End value should be greater than start value
 *  3. Start/End for L1 ratio must fall between the range of 0.01-1
 * D. For Mode combox: when selection is changed, make sure associated validation stated above is performed accordingly
 *
 * <p>
 * Copyright:   Copyright (c) 2022 IBM Inc. All Rights Reserved
 * Revisions:   Mar 23, 2022 - yle - Initial version
 */

public class ManagedAlphaValuesPlusPanel implements ManagedPanel, ControlListener {

    /**
     * @noinspection FieldCanBeLocal, UnusedDeclaration
     */
    private ManagedPanelContext context;

    /**
     * The main ExtensionObjectSwingUI for this feature.
     */
    private SwingFeatureUI extensionSwingUI;

    public ManagedAlphaValuesPlusPanel() {
        context = null;
        extensionSwingUI = null;
    }

    public void setExtensionSwingUI(ExtensionObjectSwingUI uiObject) {
        this.extensionSwingUI = (SwingFeatureUI) uiObject;
        ControlManager controlManager = this.extensionSwingUI.getControlManager();
        // Setup the ControlListeners
        Control modeControl = controlManager.getControl(MODE);
        if (modeControl != null) {
            modeControl.addControlListener(this);
        }
        // There are TWO set of alpha like values
        addOrRemoveListenersForAlphaSection(controlManager, true, true);
        addOrRemoveListenersForAlphaSection(controlManager, false, true);

        Control metricControl = controlManager.getControl(METRIC);
        if (metricControl != null) {
            metricControl.addControlListener(this);
        }
    }

    // ------------------------------- ManagedPanel Implementation ------------------------------- //

    /**
     * This is the first method to be called on the manager after
     * it has been created. This should be used to initialise member variables. Because the
     * UI is still being created, objects exposed via the context object will not be fully
     * initialised. If possible, implementations should defer calling methods on the context
     * until other SPI methods are invoked.
     *
     * @param panelId    the panel id of this panel if one was specified
     * @param properties a collection of string properties associated with the panel declaration
     * @param context    the managed panel context
     */
    public void initManagedPanel(String panelId, Properties properties, ManagedPanelContext context) {
        this.context = context;
        SwingUtilities.invokeLater(() -> setExtensionSwingUI(context.getExtensionObjectUI()));
    }

    /**
     * Creates a new ManagedUIElement for a custom ChildElement.
     *
     * @param id The value of the "id" attribute in the extension XML
     * @return A newly created ManagedUIElement or null
     */
    public ManagedUIElement createManagedUIElement(String id) {
        return null;
    }

    /**
     * Called to notify the managed panel that the window which contains the managed panel is
     * being destroyed. This provides an opportunity to free
     * any resources than need to be freed explicitly. This is the
     * last method called on this object.
     */
    public void disposeManagedPanel() {
        ControlManager controlManager = this.extensionSwingUI.getControlManager();
        Control modeControl = controlManager.getControl(MODE);
        if (modeControl != null) {
            modeControl.removeControlListener(this);
        }
        // Remove the ControlListeners
        addOrRemoveListenersForAlphaSection(controlManager, true, false);
        addOrRemoveListenersForAlphaSection(controlManager, false, false);

        Control metricControl = controlManager.getControl(METRIC);
        if (metricControl != null) {
            metricControl.removeControlListener(this);
        }
    }

    // ----------------------------- ControlListener Implementation ------------------------------- //

    /**
     * Listen when control value is changed
     */
    public void controlValueChanged(ControlEvent event) {
        // When loadingState is true, we just ignore and return, fix https://github.ibm.com/SPSS/stats_java_ui/issues/5684
        Session session = extensionSwingUI.getFeature().getSession();
        if (session instanceof CFSession && ((CFSession)session).getLoadingState())
            return;
        String propertyName = event.getPropertyName();
        boolean isL1 = propertyName.contains("L1");
        boolean isAlphaField = propertyName.equals(SINGLE_VALUES) || propertyName.equals(START) || propertyName.equals(END) || propertyName.equals(BY)
                || propertyName.equals(SINGLE_VALUES_L1) || propertyName.equals(START_L1) || propertyName.equals(END_L1) || propertyName.equals(BY_L1);

        if (propertyName.equals(SPECIFY_SINGLE) || propertyName.equals(SPECIFY_GRID)
                || propertyName.equals(SPECIFY_SINGLE_L1) || propertyName.equals(SPECIFY_GRID_L1)) {
            validateCheckboxPair(propertyName, isL1);
        } else if (isAlphaField) {
            checkValidNumbers(propertyName, isL1);
        } else if (propertyName.equals(MODE)) {
            validateCheckboxPair(SPECIFY_SINGLE_L1, true);
            validateCheckboxPair(SPECIFY_SINGLE, false);
            checkOneAlphaValue(SINGLE_VALUES_L1);
            checkOneAlphaValue(SINGLE_VALUES);
        } else if (propertyName.equals(METRIC)) {
            checkValidNumbers(START, false);
            checkValidNumbers(END, false);
            checkValidNumbers(SINGLE_VALUES, false);
        }
    }

    public void controlSelectionChanged(ControlEvent event) {
    }

    // -------------------------------------------------------------------------------------------- //
    private void addOrRemoveListenersForAlphaSection(ControlManager controlManager, boolean doL1, boolean isAdd) {
        Control specifySingleControl = controlManager.getControl(!doL1 ? SPECIFY_SINGLE : SPECIFY_SINGLE_L1);
        if (specifySingleControl != null) {
            if (isAdd)
                specifySingleControl.addControlListener(this);
            else
                specifySingleControl.removeControlListener(this);
        }
        Control specifyGridControl = controlManager.getControl(!doL1 ? SPECIFY_GRID : SPECIFY_GRID_L1);
        if (specifyGridControl != null) {
            if (isAdd)
                specifyGridControl.addControlListener(this);
            else
                specifyGridControl.removeControlListener(this);
        }
        Control singleValuesControl = controlManager.getControl(!doL1 ? SINGLE_VALUES : SINGLE_VALUES_L1);
        if (singleValuesControl != null) {
            if (isAdd)
                singleValuesControl.addControlListener(this);
            else
                singleValuesControl.removeControlListener(this);
        }
        Control startControl = controlManager.getControl(!doL1 ? START : START_L1);
        if (startControl != null) {
            if (isAdd)
                startControl.addControlListener(this);
            else
                startControl.removeControlListener(this);

        }
        Control endControl = controlManager.getControl(!doL1 ? END : END_L1);
        if (endControl != null) {
            if (isAdd)
                endControl.addControlListener(this);
            else
                endControl.removeControlListener(this);
        }
        Control byControl = controlManager.getControl(!doL1 ? BY : BY_L1);
        if (byControl != null) {
            if (isAdd)
                byControl.addControlListener(this);
            else
                byControl.removeControlListener(this);
        }
    }

    /**
     * Validation upon Specify... Checkbox selection/unselection
     * @param propertyName The current property name
     * @param isL1 true if it is L1 ratio related section, false otherwise
     */
    private void validateCheckboxPair(String propertyName, boolean isL1) {
        Boolean specifySingleChecked = (Boolean) extensionSwingUI.getControlValue(!isL1 ? SPECIFY_SINGLE : SPECIFY_SINGLE_L1);
        Control specifyGridControl = extensionSwingUI.getControlManager().getControl(!isL1 ? SPECIFY_GRID : SPECIFY_GRID_L1);
        Boolean specifyGridChecked = (Boolean) extensionSwingUI.getControlValue(!isL1 ? SPECIFY_GRID : SPECIFY_GRID_L1);
        if (!specifySingleChecked && (!specifyGridControl.isEnabled() || !specifyGridChecked)) {
            AlertOptionPane.showErrorMessageDialog(extensionSwingUI.getRootComponent(),
                    extensionSwingUI.getSwingResourceProvider().getString("single_grid_alpha_select_error.MANAGED"),
                    getUISession().getApplication().getApplicationBranding().getApplicationName());
            Control ctrl = extensionSwingUI.getControlManager().getControl(propertyName);
            // Determine which one to check after the error alert
            String targetCtrlName = propertyName;
            String specifySingle = isL1 ? SPECIFY_SINGLE_L1 : SPECIFY_SINGLE;
            String specifyGrid = isL1 ? SPECIFY_GRID_L1 : SPECIFY_GRID;
            if (propertyName.equals(specifySingle) && specifyGridControl.isEnabled())
                targetCtrlName = specifyGrid;
            else if (propertyName.equals(specifyGrid))
                targetCtrlName = specifySingle;
            // Remove listener before reset the value
            ctrl.removeControlListener(this);
            extensionSwingUI.getControlManager().getControl(targetCtrlName).setControlValue(targetCtrlName, true);
            ctrl.addControlListener(this);
        }
        // Need to re-check duplicates here because checkbox state change causes textbox enable/disable to change
        handleDuplicates(SINGLE_VALUES, true);
        handleDuplicates(SINGLE_VALUES_L1, false);
    }

    /**
     * Check to validate the field which was just clicked away
     *
     * @param propertyName The current propertyName to be clicked away
     * @param isL1 true if the active control is within L1 ratio related section
     */
    private void checkValidNumbers(String propertyName, boolean isL1) {
        String single = !isL1 ? SINGLE_VALUES : SINGLE_VALUES_L1;
        String start = !isL1 ? START : START_L1;
        String end = !isL1 ? END : END_L1;
        String by = !isL1 ? BY : BY_L1;

        // check if current value is all numeric, or positive when required
        checkCurrentValue(propertyName);

        Double startValue = (Double) extensionSwingUI.getControlValue(start);
        Double endValue = (Double) extensionSwingUI.getControlValue(end);
        Double byValue = (Double) extensionSwingUI.getControlValue(by);
        // Check Start and End value to validate
        if (propertyName.equals(start) || propertyName.equals(end) || propertyName.equals(by)) {
            if (startValue != null && endValue != null && byValue != null && endValue <= startValue) {
                AlertOptionPane.showErrorMessageDialog(extensionSwingUI.getRootComponent(),
                        extensionSwingUI.getSwingResourceProvider().getString("end_less_than_start_error.MANAGED"),
                        getUISession().getApplication().getApplicationBranding().getApplicationName());
                // Switch the start and end value
                ControlManager cm = extensionSwingUI.getControlManager();
                Control startCtrl = cm.getControl(start);
                Control endCtrl = cm.getControl(end);
                startCtrl.removeControlListener(this);
                endCtrl.removeControlListener(this);
                cm.getControl(start).setControlValue(start, endValue);
                cm.getControl(end).setControlValue(end, startValue);
                startCtrl.addControlListener(this);
                endCtrl.addControlListener(this);
                UIUtilities.getInstance().requestFocusForControl(cm.getControl(start));
            }
        }
        if (propertyName.equals(single)) {
            checkOneAlphaValue(single);
        }
        // Handle duplicates
        handleDuplicates(propertyName, isL1);
    }

    /**
     * Check SingleValues field to make sure that when Mode=Fit, only one value is permitted
     * @param singlePropName property name for single field, could be "singleValues" or "singleValuesL1"
     */
    private void checkOneAlphaValue(String singlePropName) {
        CFEnum m = (CFEnum) extensionSwingUI.getControlValue(MODE);
        Object singleValue = extensionSwingUI.getControlValue(singlePropName);
        String[] valArray = getSingleValues(singleValue.toString());
        // When Mode=Fit: Only one value of alpha is allowed
        if (m.toString().equals(MODE_FIT)) {
            if (valArray.length > 1) {
                showErrorAndUpdate(singlePropName, valArray[0], "only_one_alpha_value_error.MANAGED");
            }
        }
    }

    /**
     * Check for duplicate values - between singleValues and values specified in grid
     */
    private boolean checkDuplicate(String propertyName, Double[] doubleValues, Double startValue, Double endValue,
                                   Double byValue, Set<Double> resultSet, boolean isL1) {
        boolean hasDuplicate = false;

        Set<Double> set = new HashSet<>();
        if (propertyName.equals(!isL1 ? SINGLE_VALUES : SINGLE_VALUES_L1)) {
            // check duplicate within single values
            List<Double> list = Arrays.asList(doubleValues);
            boolean hasDuplicate1 = false;
            for (Double l : list) {
                if (!set.add(l)) {
                    hasDuplicate1 = true;
                } else {
                    resultSet.add(l);
                }
            }

            ControlManager cm = extensionSwingUI.getControlManager();
            Control startControl = cm.getControl(!isL1 ? START : START_L1);
            // Check duplication comparing with grid values
            boolean hasDuplicate2 = false;
            if (startControl.isEnabled() && startValue != null && endValue != null && byValue != null) {
                hasDuplicate2 = hasDuplicateAgainstGrid(doubleValues, startValue, endValue, byValue, resultSet);
            }
            hasDuplicate = hasDuplicate1 | hasDuplicate2;
        } else {
            resultSet.addAll(Arrays.asList(doubleValues));
            hasDuplicate = hasDuplicateAgainstGrid(doubleValues, startValue, endValue, byValue, resultSet);
        }
        return hasDuplicate;
    }

    /**
     * Check duplication between singleValues against all values defined by grid
     *
     * @return true if duplication is found, false otherwise
     */
    private boolean hasDuplicateAgainstGrid(Double[] doubleValues, Double startValue, Double endValue, Double byValue, Set<Double> resultSet) {
        boolean hasDuplicate = false;
        if (startValue != null && endValue != null && byValue != null) {
            ArrayList<Double> all = getAllParamValuesInGrid(startValue, endValue, byValue);
            // Further remove the duplicates against grid
            for (Double v : doubleValues) {
                if (all.contains(v)) {
                    hasDuplicate = true;
                    resultSet.remove(v);
                }
            }
        }
        return hasDuplicate;
    }

    /**
     * Handle duplication either within single values field or between single values field against grid values
     * @param propertyName The current property name
     * @param isL1 true if the active control is within L1 ratio section
     */
    private void handleDuplicates(String propertyName, boolean isL1) {
        String single = !isL1 ? SINGLE_VALUES : SINGLE_VALUES_L1;
        Object singleValue = extensionSwingUI.getControlValue(single);
        Control singleCtrl = extensionSwingUI.getControlManager().getControl(!isL1 ? SINGLE_VALUES : SINGLE_VALUES_L1);
        Double startValue = (Double) extensionSwingUI.getControlValue(!isL1 ? START : START_L1);
        Double endValue = (Double) extensionSwingUI.getControlValue(!isL1 ? END : END_L1);
        Double byValue = (Double) extensionSwingUI.getControlValue(!isL1 ? BY : BY_L1);
        if ((startValue == null || endValue == null || byValue == null) && (singleValue == null || singleValue.toString().length() == 0))
            return;

        Double[] doubleValues = null;
        String[] valArray = null;
        if (singleCtrl.isEnabled()) {
            // ignore all trailing or heading white spaces
            valArray = getSingleValues(singleValue.toString());
            doubleValues = Arrays.stream(valArray).map(Double::valueOf).toArray(Double[]::new);
        }

        Set<Double> resultSet = new HashSet<>();
        Set<String> temp = new HashSet<>();
        StringBuilder builder = new StringBuilder();
        if (doubleValues != null && checkDuplicate(propertyName, doubleValues, startValue, endValue, byValue, resultSet, isL1)) {
            // remove duplicates from user input to maintain the original input
            for (String v : valArray) {
                if (resultSet.contains(Double.valueOf(v))) {
                    if (temp.add(v)) {
                        builder.append(v);
                        builder.append(" ");
                    }
                }
            }
            // Duplicate values have been removed
            String newValues = builder.toString().trim();
            showErrorAndUpdate(single, newValues, "alpha_duplicate_value_error.MANAGED");
        }
    }

    /**
     * Turn string value of individual values to an array of String
     * @param strValue The string value of the field
     * @return array of String
     */
    private String[] getSingleValues(String strValue) {
        // ignore all trailing or heading white spaces
        return strValue.replaceAll("(^\\s+|\\s+$)", "").split("\\s+");
    }

    /**
     * Check current control value to see if the value is empty or numeric values, or positive values if required
     *
     * @param propertyName the current propertyName in concern
     */
    private void checkCurrentValue(String propertyName) {
        Object val = extensionSwingUI.getControlValue(propertyName);

        // Note: For number field, we would get either numeric value or null, won't get empty string
        // Note: Empty field checking resides in extension.xml, we will ignore here
        if (val == null || val.toString().trim().length() == 0) return;

        CFEnum metricVal = (CFEnum)extensionSwingUI.getControlValue(METRIC);
        boolean isLinear = metricVal.toString().equals(LINEAR);
        // For number field, we don't need to check here because the CDB ui already guards it
        if ((propertyName.equals(SINGLE_VALUES) || propertyName.equals(SINGLE_VALUES_L1)) && val instanceof String) {
            boolean isAlpha = propertyName.equals(SINGLE_VALUES);
            char decimalSep = new DecimalFormatSymbols(UIToolResUtil.getSPSSLocale()).getDecimalSeparator();
            String v = val.toString().trim();
            StringBuilder builder = new StringBuilder();
            // Check if all numbers, or decimal separator, or space for singleValues field
            boolean hasInvalid = false;
            for (int i = 0; i < v.length(); i++) {
                char ch = v.charAt(i);
                if (Character.isDigit(ch) || ch == decimalSep || ch == ' ' || ((!isAlpha || !isLinear) && ch == '-') ) {
                    builder.append(ch);
                } else {
                    hasInvalid = true;
                }
            }
            if (hasInvalid) {
                String msg = isLinear ? "alpha_numeric_positive_value_error.MANAGED" : "alpha_numeric_value_error.MANAGED";
                showErrorAndUpdate(propertyName, builder.toString(), msg);
            }
            if (!isAlpha) {
                // all numeric values when gets here. Check if the value is within the range for L1 ratio (0.01 - 1)
                String[] values = getSingleValues(val.toString());
                boolean otr = false;
                StringBuilder builder2 = new StringBuilder();
                for (String sv : values) {
                    double dv = Double.parseDouble(sv);
                    if (dv < RATIO_MIN || dv > RATIO_MAX) {
                        otr = true;
                    } else {
                        builder2.append(dv);
                        builder2.append(" ");
                    }
                }
                if (otr)
                    showErrorAndUpdate(propertyName, builder2.toString().trim(), "ratio_numeric_range_value_error.MANAGED");
            }
            checkOneAlphaValue(propertyName);

        } else if (isLinear && (propertyName.equals(START) || propertyName.equals(END)) ) {
            // For Alpha Linear only: Check if the value should be positive only
            Double value = (Double) extensionSwingUI.getControlValue(propertyName);
            if (value < 0) {
                showErrorAndUpdate(propertyName, String.valueOf(Math.abs(value)), "alpha_positive_value_error.MANAGED");
            }
        } else if (propertyName.equals(START_L1) || propertyName.equals(END_L1)) {
            double dVal = (Double)val;
            if (dVal < RATIO_MIN || dVal > RATIO_MAX) {
                double fb = dVal;
                if (dVal < 0)
                    fb = Math.abs(dVal);
                double fallback = (fb > RATIO_MIN && fb < RATIO_MAX)? fb : (propertyName.equals(START_L1) ? RATIO_MIN : RATIO_MAX);
                showErrorAndUpdate(propertyName, fallback, "ratio_numeric_range_value_error.MANAGED");
            }
        }
    }

    /**
     * Get all the param values defined by grid
     *
     * @param start The start value
     * @param end   The end value
     * @param by    The by value
     * @return the ArrayList of Double containing all those values
     */
    private ArrayList<Double> getAllParamValuesInGrid(Double start, Double end, Double by) {
        ArrayList<Double> list = new ArrayList<>();

        for (int i = 0; i <= 50; i++) {
            // Using BigDecimal to fix issue: https://github.ibm.com/SPSS/stats_defects/issues/2038
            // Because the drawback of primitive type float or double number's calculation resulting
            // unexpected number sometimes, for example, 0.1*3 being 0.30000000000000004 instead of just 0.3,
            // making checking for duplication fails, Use BigDecimal solves this issue
            BigDecimal bs = new BigDecimal(String.valueOf(start));
            BigDecimal bb = new BigDecimal(String.valueOf(by));
            BigDecimal bi = new BigDecimal(String.valueOf(i));
            double result = bi.multiply(bb).add(bs).doubleValue();
            if (result <= end) {
                list.add(result);
            } else
                break;
        }
        return list;
    }

    /**
     * Obtain the UISession
     *
     * @return the UISession
     */
    private UISession getUISession() {
        return extensionSwingUI.getSwingResourceProvider().getUISession();
    }

    /**
     * The Common method to show error alert and update with valid values, and move focus to the control
     *
     * @param propertyName The current propertyName in concern
     * @param newVal       The new valid value Object to set, could be Double, String, Integer etc.
     * @param errorKey     The error message id
     */
    private void showErrorAndUpdate(String propertyName, Object newVal, String errorKey) {
        // Show error message
        AlertOptionPane.showErrorMessageDialog(extensionSwingUI.getRootComponent(),
                extensionSwingUI.getSwingResourceProvider().getString(errorKey),
                getUISession().getApplication().getApplicationBranding().getApplicationName());

        ControlManager cm = extensionSwingUI.getControlManager();
        Control currentCtrl = cm.getControl(propertyName);
        currentCtrl.removeControlListener(this);
        cm.getControl(propertyName).setControlValue(propertyName, newVal);
        currentCtrl.addControlListener(this);
        UIUtilities.getInstance().requestFocusForControl(cm.getControl(propertyName));
    }

    // IDs of controls
    private final static String MODE = "mode";
    // L1 ratio section
    private final static String SPECIFY_SINGLE_L1 = "singleL1Check";
    private final static String SPECIFY_GRID_L1 = "gridL1Check";
    private final static String SINGLE_VALUES_L1 = "singleValuesL1";

    private final static String START_L1 = "startL1";
    private final static String END_L1 = "endL1";
    private final static String BY_L1 = "byL1";

    // Alpha section
    private final static String SPECIFY_SINGLE = "singleCheck";
    private final static String SPECIFY_GRID = "gridCheck";
    private final static String SINGLE_VALUES = "singleValues";

    private final static String START = "start";
    private final static String END = "end";
    private final static String BY = "by";

    private final static String METRIC = "metric";

    // Mode combo selection value - first selection
    private final static String MODE_FIT = "mode1";
    // Metric radio group selection - first radio selection
    private final static String LINEAR = "linearRadio";

    private final static double RATIO_MIN = 0.01;
    private final static double RATIO_MAX = 1;
}